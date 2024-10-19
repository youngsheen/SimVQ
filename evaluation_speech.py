import os
import sys
sys.path.append(os.getcwd())
import glob
from metrics.UTMOS import UTMOSScore
from metrics.periodicity import calculate_periodicity_metrics
import torchaudio
from pesq import pesq
import numpy as np
import torch
import math
from pystoi import stoi
from pathlib import Path
from tqdm import tqdm

import importlib
from omegaconf import OmegaConf
import argparse

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_config(config_path, display=False):
    config = OmegaConf.load(config_path)
    if display:
        print(yaml.dump(OmegaConf.to_container(config)))
    return config

def load_vqgan_new(config, ckpt_path=None, is_gumbel=False):
    model = instantiate_from_config(config.model)
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
    return model.eval()


def get_obj_from_str(string, reload=False):
    print(string)
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not "class_path" in config:
        raise KeyError("Expected key `class_path` to instantiate.")
    return get_obj_from_str(config["class_path"])(**config.get("init_args", dict()))


def main(args):
    config_data = OmegaConf.load(args.config_file)
    config_data.data.init_args.batch_size = args.batch_size

    config_model = load_config(args.config_file, display=False)
    model = load_vqgan_new(config_model, ckpt_path=args.ckpt_path).to(DEVICE)
    codebook_size = model.quantize.n_e
    
    dataset = instantiate_from_config(config_data.data)
    dataset.prepare_data()
    dataset.setup()
    
    usage = {}
    for i in range(codebook_size):
        usage[i] = 0
        
    paths = []
    
    with torch.no_grad():
        for batch in tqdm(dataset._test_dataloader()):
            assert batch["waveform"].shape[0] == 1
            paths.append(batch["audio_path"][0])
            
            
            audio = batch["waveform"].to(DEVICE)
        
            if model.use_ema:
                with model.ema_scope():
                    quant, diff, indices, _ = model.encode(audio)
                    reconstructed_audios = model.decode(quant)
            else:
               quant, diff, indices, _ = model.encode(audio)
               reconstructed_audios = model.decode(quant)
               
            for index in indices.flatten():
                usage[index.item()] += 1
                

            audio_path = args.ckpt_path.parent / "recons" / batch["audio_path"][0]
            audio_path.parent.mkdir(parents=True, exist_ok=True)
            torchaudio.save(audio_path.as_posix(), reconstructed_audios[0].cpu().clip(min=-0.99, max=0.99), sample_rate=24000, encoding='PCM_S', bits_per_sample=16)
            
            
    num_count = sum([1 for key, value in usage.items() if value > 0])
    utilization = num_count / codebook_size
    
    UTMOS=UTMOSScore(device=DEVICE)

    utmos_sumgt=0
    utmos_sumencodec=0
    pesq_sumpre=0
    f1score_sumpre=0
    stoi_sumpre=[]
    f1score_filt=0

    for i in tqdm(range(len(paths))):
        rawwav,rawwav_sr=torchaudio.load(os.path.join(os.environ['DATA_ROOT'], paths[i]))
        prewav,prewav_sr=torchaudio.load((args.ckpt_path.parent / "recons" / paths[i]).as_posix())
        
        rawwav=rawwav.to(DEVICE)
        prewav=prewav.to(DEVICE)
   
        rawwav_16k=torchaudio.functional.resample(rawwav, orig_freq=rawwav_sr, new_freq=16000)  #测试UTMOS的时候必须重采样
        prewav_16k=torchaudio.functional.resample(prewav, orig_freq=prewav_sr, new_freq=16000)

        # 1.UTMOS
        print("****UTMOS_raw",i,UTMOS.score(rawwav_16k.unsqueeze(1))[0].item())
        print("****UTMOS_encodec",i,UTMOS.score(prewav_16k.unsqueeze(1))[0].item())
        utmos_sumgt+=UTMOS.score(rawwav_16k.unsqueeze(1))[0].item()
        utmos_sumencodec+=UTMOS.score(prewav_16k.unsqueeze(1))[0].item()
    
        # breakpoint()

        ## 2.PESQ  
        min_len=min(rawwav_16k.size()[1],prewav_16k.size()[1])
        rawwav_16k_pesq=rawwav_16k[:,:min_len].squeeze(0)
        prewav_16k_pesq=prewav_16k[:,:min_len].squeeze(0)
        pesq_score = pesq(16000, rawwav_16k_pesq.cpu().numpy(), prewav_16k_pesq.cpu().numpy(), "wb", on_error=1)
        print("****PESQ",i,pesq_score)
        pesq_sumpre+=pesq_score
        # breakpoint()

        ## 3.F1-score
        min_len=min(rawwav_16k.size()[1],prewav_16k.size()[1])
        rawwav_16k_f1score=rawwav_16k[:,:min_len]
        prewav_16k_f1score=prewav_16k[:,:min_len]
        periodicity_loss, pitch_loss, f1_score = calculate_periodicity_metrics(rawwav_16k_f1score,prewav_16k_f1score)
        print("****f1",periodicity_loss, pitch_loss, f1_score,f1score_sumpre)
        if(math.isnan(f1_score)):
            f1score_filt+=1
            print("*****",f1score_filt)
        else:
            f1score_sumpre+=f1_score
        # breakpoint()


        ## 4.STOI
        # for ljspeech
        # rawwav_24k=torchaudio.functional.resample(rawwav, orig_freq=rawwav_sr, new_freq=24000)
        # min_len=min(rawwav_24k.size()[1],prewav.size()[1])
        # rawwav_stoi=rawwav_24k[:,:min_len].squeeze(0)
        # prewav_stoi=prewav[:,:min_len].squeeze(0)
        # tmp_stoi=stoi(rawwav_stoi.cpu(),prewav_stoi.cpu(),24000,extended=False)
        # print("****stoi",tmp_stoi)
        # stoi_sumpre.append(tmp_stoi)
        # # breakpoint()

        min_len=min(rawwav.size()[1],prewav.size()[1])
        rawwav_stoi=rawwav[:,:min_len].squeeze(0)
        prewav_stoi=prewav[:,:min_len].squeeze(0)
        tmp_stoi=stoi(rawwav_stoi.cpu(),prewav_stoi.cpu(),rawwav_sr,extended=False)
        print("****stoi",tmp_stoi)
        stoi_sumpre.append(tmp_stoi)
        
    def print_and_save(message, file):
        print(message)  
        file.write(message + '\n') 
        
    with open(Path(args.ckpt_path).parent / "result.txt", 'w') as f:
        print_and_save(f"UTMOS_raw: {utmos_sumgt}, {utmos_sumgt/len(paths)}", f)
        print_and_save(f"UTMOS_encodec: {utmos_sumgt}, {utmos_sumencodec/len(paths)}", f)
        print_and_save(f"PESQ: {pesq_sumpre}, {pesq_sumpre/len(paths)}", f)
        print_and_save(f"F1_score: {f1score_sumpre}, {f1score_sumpre/(len(paths)-f1score_filt)}, {f1score_filt}", f)
        print_and_save(f"STOI: {np.mean(stoi_sumpre)}", f)
        print_and_save(f"utilization: {utilization}", f)
    
    
def get_args():
    parser = argparse.ArgumentParser(description="inference parameters")
    parser.add_argument("--config_file", required=True, type=str)
    parser.add_argument("--ckpt_path", required=True, type=Path)
    parser.add_argument("--batch_size", default=1, type=int)

    return parser.parse_args()


if __name__=="__main__":
    args = get_args()
    main(args)