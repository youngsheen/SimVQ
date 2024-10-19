import torch
import torch.nn as nn
import torch.nn.functional as F
import typing as tp
from taming.modules.discriminator.mpmr import MultiPeriodDiscriminator, MultiResolutionDiscriminator
from taming.modules.discriminator.dac import DACDiscriminator
from .speech_loss import MelSpecReconstructionLoss, GeneratorLoss, DiscriminatorLoss, FeatureMatchingLoss, DACGANLoss


class VQSTFTWithDiscriminator(nn.Module):
    def __init__(self, disc_start, codebook_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=1, disc_factor=1.0, disc_weight=1.0,
                 commit_weight = 0.25, codebook_enlarge_ratio=3, codebook_enlarge_steps=2000,
                 perceptual_weight=1.0, use_actnorm=False, disc_conditional=False,
                 disc_loss="hinge", gen_loss_weight=None, lecam_loss_weight=None, 
                 mel_loss_coeff=None, mrd_loss_coeff=None,
                 sample_rate=None):
        super().__init__()
        assert disc_loss in ["hinge", "vanilla", "non_saturate"]
        self.codebook_weight = codebook_weight
        self.commit_weight = commit_weight
        self.codebook_enlarge_ratio = codebook_enlarge_ratio
        self.codebook_enlarge_steps = codebook_enlarge_steps
        self.gen_loss_weight = gen_loss_weight
        
        self.mel_loss_coeff=mel_loss_coeff
        self.mrd_loss_coeff=mrd_loss_coeff
        
        self.multiperioddisc = MultiPeriodDiscriminator()
        self.multiresddisc = MultiResolutionDiscriminator()
        self.dac = DACDiscriminator()
        self.dacdiscriminator = DACGANLoss(self.dac)
        self.disc_loss = DiscriminatorLoss()
        self.gen_loss = GeneratorLoss()
        self.feat_matching_loss = FeatureMatchingLoss()
        self.melspec_loss = MelSpecReconstructionLoss(sample_rate=sample_rate)
        
        self.discriminator_iter_start = disc_start
        
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
                    
    def forward(self, codebook_loss, loss_break, inputs, reconstructions, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train"):
        
        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            
            loss_dac_1, loss_dac_2 = self.dacdiscriminator.generator_loss(reconstructions, inputs)
            
            _, gen_score_mp, fmap_rs_mp, fmap_gs_mp = self.multiperioddisc(
                    y=inputs, y_hat=reconstructions
                )
            _, gen_score_mrd, fmap_rs_mrd, fmap_gs_mrd = self.multiresddisc(
                    y=inputs, y_hat=reconstructions
                )
            loss_gen_mp, list_loss_gen_mp = self.gen_loss(disc_outputs=gen_score_mp)
            loss_gen_mrd, list_loss_gen_mrd = self.gen_loss(disc_outputs=gen_score_mrd)
            loss_gen_mp = loss_gen_mp / len(list_loss_gen_mp)
            loss_gen_mrd = loss_gen_mrd / len(list_loss_gen_mrd)
            loss_fm_mp = self.feat_matching_loss(fmap_r=fmap_rs_mp, fmap_g=fmap_gs_mp) / len(fmap_rs_mp)
            loss_fm_mrd = self.feat_matching_loss(fmap_r=fmap_rs_mrd, fmap_g=fmap_gs_mrd) / len(fmap_rs_mrd)

            mel_loss = self.melspec_loss(reconstructions, inputs)
            rec_loss = mel_loss
            
            loss = (
                self.gen_loss_weight * (loss_gen_mp
                + self.mrd_loss_coeff * loss_gen_mrd
                + loss_fm_mp
                + self.mrd_loss_coeff * loss_fm_mrd
                + loss_dac_1
                + loss_dac_2)
                + self.mel_loss_coeff * mel_loss
                + self.commit_weight * loss_break.commitment
            )

            log = {"{}/total_loss".format(split): loss.clone().detach(),
                    "{}/commit_loss".format(split): loss_break.commitment.detach(),
                    "{}/reconstruct_loss".format(split): rec_loss.detach(),
                    "{}/multi_period_loss".format(split): loss_gen_mp.detach(),
                    "{}/multi_res_loss".format(split): loss_gen_mrd.detach(),
                    "{}/feature_matching_mp".format(split): loss_fm_mp.detach(),
                    "{}/feature_matching_mrd".format(split): loss_fm_mrd.detach(),
                    "{}/loss_dac_1".format(split): loss_dac_1.detach(),
                    "{}/loss_dac_2".format(split): loss_dac_2.detach(),
                    }

            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            
            loss_dac = self.dacdiscriminator.discriminator_loss(reconstructions.contiguous().detach(), inputs.contiguous().detach())

            real_score_mp, gen_score_mp, _, _ = self.multiperioddisc(y=inputs.contiguous().detach(), y_hat=reconstructions.contiguous().detach())
            real_score_mrd, gen_score_mrd, _, _ = self.multiresddisc(y=inputs.contiguous().detach(), y_hat=reconstructions.contiguous().detach())
            loss_mp, loss_mp_real, _ = self.disc_loss(
                disc_real_outputs=real_score_mp, disc_generated_outputs=gen_score_mp
            )
            loss_mrd, loss_mrd_real, _ = self.disc_loss(
                disc_real_outputs=real_score_mrd, disc_generated_outputs=gen_score_mrd
            )
            loss_mp /= len(loss_mp_real)
            loss_mrd /= len(loss_mrd_real)
            loss = loss_mp + self.mrd_loss_coeff * loss_mrd + loss_dac
            
            log = {"{}/total_loss".format(split): loss.detach(),
                   "{}/multi_res_loss".format(split): loss_mrd.detach(),
                   "{}/multi_period_loss".format(split): loss_mp.detach(),
                   "{}/dac".format(split): loss_dac.detach(),
                   }
            return loss, log