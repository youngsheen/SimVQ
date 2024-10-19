import os, tarfile, glob, shutil
import yaml
import numpy as np
from tqdm import tqdm
from PIL import Image
import albumentations
from omegaconf import OmegaConf
from torch.utils.data import Dataset

from taming.data.base import ImagePaths
from taming.util import download, retrieve
import taming.data.utils as bdu


def give_synsets_from_indices(indices, path_to_yaml="data/imagenet_idx_to_synset.yaml"):
    synsets = []
    with open(path_to_yaml) as f:
        di2s = yaml.load(f)
    for idx in indices:
        synsets.append(str(di2s[idx]))
    print("Using {} different synsets for construction of Restriced Imagenet.".format(len(synsets)))
    return synsets


def str_to_indices(string):
    """Expects a string in the format '32-123, 256, 280-321'"""
    assert not string.endswith(","), "provided string '{}' ends with a comma, pls remove it".format(string)
    subs = string.split(",")
    indices = []
    for sub in subs:
        subsubs = sub.split("-")
        assert len(subsubs) > 0
        if len(subsubs) == 1:
            indices.append(int(subsubs[0]))
        else:
            rang = [j for j in range(int(subsubs[0]), int(subsubs[1]))]
            indices.extend(rang)
    return sorted(indices)


class ImageNetBase(Dataset):
    def __init__(self, config=None):
        self.config = config or OmegaConf.create()
        if not type(self.config)==dict:
            self.config = OmegaConf.to_container(self.config)
        self._prepare()
        self._prepare_synset_to_human()
        self._prepare_idx_to_synset()
        self._load()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def _prepare(self):
        raise NotImplementedError()

    def _filter_relpaths(self, relpaths):
        ignore = set([
            "n06596364_9591.JPEG",
        ])
        relpaths = [rpath for rpath in relpaths if not rpath.split("/")[-1] in ignore]
        if "sub_indices" in self.config:
            indices = str_to_indices(self.config["sub_indices"])
            synsets = give_synsets_from_indices(indices, path_to_yaml=self.idx2syn)  # returns a list of strings
            files = []
            for rpath in relpaths:
                syn = rpath.split("/")[0]
                if syn in synsets:
                    files.append(rpath)
            return files
        else:
            return relpaths

    def _prepare_synset_to_human(self):
        SIZE = 2655750
        URL = "https://heibox.uni-heidelberg.de/f/9f28e956cd304264bb82/?dl=1"
        self.human_dict = os.path.join(self.root, "synset_human.txt")
        if (not os.path.exists(self.human_dict) or
                not os.path.getsize(self.human_dict)==SIZE):
            download(URL, self.human_dict)

    def _prepare_idx_to_synset(self):
        URL = "https://heibox.uni-heidelberg.de/f/d835d5b6ceda4d3aa910/?dl=1"
        self.idx2syn = os.path.join(self.root, "imagenet_idx_to_synset.yaml")
        if (not os.path.exists(self.idx2syn)):
            download(URL, self.idx2syn)

    def _load(self):
        with open(self.txt_filelist, "r") as f:
            self.relpaths = f.read().splitlines()
            l1 = len(self.relpaths)
            self.relpaths = self._filter_relpaths(self.relpaths)
            print("Removed {} files from filelist during filtering.".format(l1 - len(self.relpaths)))

        self.synsets = [p.split("/")[0] for p in self.relpaths]
        self.abspaths = [os.path.join(self.datadir, p) for p in self.relpaths]

        unique_synsets = np.unique(self.synsets)
        class_dict = dict((synset, i) for i, synset in enumerate(unique_synsets))
        self.class_labels = [class_dict[s] for s in self.synsets]

        with open(self.human_dict, "r") as f:
            human_dict = f.read().splitlines()
            human_dict = dict(line.split(maxsplit=1) for line in human_dict)

        self.human_labels = [human_dict[s] for s in self.synsets]

        labels = {
            "relpath": np.array(self.relpaths),
            "synsets": np.array(self.synsets),
            "class_label": np.array(self.class_labels),
            "human_label": np.array(self.human_labels),
        }
        self.data = ImagePaths(self.abspaths,
                               labels=labels,
                               size=retrieve(self.config, "size", default=0),
                               random_crop=self.random_crop)


class ImageNetTrain(ImageNetBase):
    NAME = "train"
    URL = "http://www.image-net.org/challenges/LSVRC/2012/"
    AT_HASH = "a306397ccf9c2ead27155983c254227c0fd938e2"
    FILES = [
        "ILSVRC2012_img_train.tar",
    ]
    SIZES = [
        147897477120,
    ]

    def _prepare(self):
        self.random_crop = retrieve(self.config, "ImageNetTrain/random_crop",
                                    default=True)
        cachedir = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("../../data/imagenet")) #specfy the path
        self.root = os.path.join(cachedir, self.NAME)
        self.datadir = self.root
        if self.config["subset"] is not None: # for training in subset
            self.txt_filelist = os.path.join("../../data", "{}_{}.txt".format(self.NAME, self.config["subset"]))
        else:
            self.txt_filelist = os.path.join(self.root, "filelist.txt")
        self.expected_length = 1281167
        if not bdu.is_prepared(self.root):
            # prep
            print("Preparing dataset {} in {}".format(self.NAME, self.root))

            datadir = self.datadir
            if not os.path.exists(datadir):
                path = os.path.join(self.root, self.FILES[0])
                if not os.path.exists(path) or not os.path.getsize(path)==self.SIZES[0]:
                    import academictorrents as at
                    atpath = at.get(self.AT_HASH, datastore=self.root)
                    assert atpath == path

                print("Extracting {} to {}".format(path, datadir))
                os.makedirs(datadir, exist_ok=True)
                with tarfile.open(path, "r:") as tar:
                    tar.extractall(path=datadir)

                print("Extracting sub-tars.")
                subpaths = sorted(glob.glob(os.path.join(datadir, "*.tar")))
                for subpath in tqdm(subpaths):
                    subdir = subpath[:-len(".tar")]
                    os.makedirs(subdir, exist_ok=True)
                    with tarfile.open(subpath, "r:") as tar:
                        tar.extractall(path=subdir)

            filelist = glob.glob(os.path.join(datadir, "**", "*.JPEG"))
            filelist = [os.path.relpath(p, start=datadir) for p in filelist]
            filelist = sorted(filelist)
            filelist = "\n".join(filelist)+"\n"
            with open(self.txt_filelist, "w") as f:
                f.write(filelist)

            bdu.mark_prepared(self.root)


class ImageNetValidation(ImageNetBase):
    NAME = "val"
    URL = "http://www.image-net.org/challenges/LSVRC/2012/"
    AT_HASH = "5d6d0df7ed81efd49ca99ea4737e0ae5e3a5f2e5"
    VS_URL = "https://heibox.uni-heidelberg.de/f/3e0f6e9c624e45f2bd73/?dl=1"
    FILES = [
        "ILSVRC2012_img_val.tar",
        "validation_synset.txt",
    ]
    SIZES = [
        6744924160,
        1950000,
    ]

    def _prepare(self):
        self.random_crop = retrieve(self.config, "ImageNetValidation/random_crop",
                                    default=False)
        cachedir = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("../../data/imagenet"))
        self.root = os.path.join(cachedir, self.NAME)
        self.datadir = self.root
        if self.config["subset"] is not None: # for debugging
            self.txt_filelist = os.path.join("../../data", "{}_{}.txt".format(self.NAME, self.config["subset"]))
        else:
            self.txt_filelist = os.path.join(self.root, "filelist.txt")

        self.expected_length = 50000
        if not bdu.is_prepared(self.root):
            # prep
            print("Preparing dataset {} in {}".format(self.NAME, self.root))

            datadir = self.datadir
            if not os.path.exists(datadir):
                path = os.path.join(self.root, self.FILES[0])
                if not os.path.exists(path) or not os.path.getsize(path)==self.SIZES[0]:
                    import academictorrents as at
                    atpath = at.get(self.AT_HASH, datastore=self.root)
                    assert atpath == path

                print("Extracting {} to {}".format(path, datadir))
                os.makedirs(datadir, exist_ok=True)
                with tarfile.open(path, "r:") as tar:
                    tar.extractall(path=datadir)

                vspath = os.path.join(self.root, self.FILES[1])
                if not os.path.exists(vspath) or not os.path.getsize(vspath)==self.SIZES[1]:
                    download(self.VS_URL, vspath)

                with open(vspath, "r") as f:
                    synset_dict = f.read().splitlines()
                    synset_dict = dict(line.split() for line in synset_dict)

                print("Reorganizing into synset folders")
                synsets = np.unique(list(synset_dict.values()))
                for s in synsets:
                    os.makedirs(os.path.join(datadir, s), exist_ok=True)
                for k, v in synset_dict.items():
                    src = os.path.join(datadir, k)
                    dst = os.path.join(datadir, v)
                    shutil.move(src, dst)

            filelist = glob.glob(os.path.join(datadir, "**", "*.JPEG"))
            filelist = [os.path.relpath(p, start=datadir) for p in filelist]
            filelist = sorted(filelist)
            filelist = "\n".join(filelist)+"\n"
            with open(self.txt_filelist, "w") as f:
                f.write(filelist)

            bdu.mark_prepared(self.root)


def get_preprocessor(size=None, random_crop=False, additional_targets=None,
                     crop_size=None):
    if size is not None and size > 0:
        transforms = list()
        rescaler = albumentations.SmallestMaxSize(max_size = size)
        transforms.append(rescaler)
        if not random_crop:
            cropper = albumentations.CenterCrop(height=size,width=size)
            transforms.append(cropper)
        else:
            cropper = albumentations.RandomCrop(height=size,width=size)
            transforms.append(cropper)
            flipper = albumentations.HorizontalFlip()
            transforms.append(flipper)
        preprocessor = albumentations.Compose(transforms,
                                              additional_targets=additional_targets)
    elif crop_size is not None and crop_size > 0:
        if not random_crop:
            cropper = albumentations.CenterCrop(height=crop_size,width=crop_size)
        else:
            cropper = albumentations.RandomCrop(height=crop_size,width=crop_size)
        transforms = [cropper]
        preprocessor = albumentations.Compose(transforms,
                                              additional_targets=additional_targets)
    else:
        preprocessor = lambda **kwargs: kwargs
    return preprocessor
