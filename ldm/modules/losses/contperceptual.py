import torch
import torch.nn as nn

from taming.modules.losses.vqperceptual import *  # TODO: taming dependency yes/no?
#from taming.modules.discriminator.model import NLayerDiscriminator, weights_init


class LPIPSWithDiscriminator(nn.Module):
    '''
    LPIPS = Learned Perceptual Image Patch Similarity, arxiv.org/pdf/1801.03924.pdf
    '''
    def __init__(self, disc_start, logvar_init=0.0, kl_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
                 perceptual_weight=1.0, use_actnorm=False, disc_conditional=False,
                 disc_loss="hinge"):
        ##import ipdb; ipdb.set_trace()
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval() # TODO, 感知loss, lpips.py in taming transformer git repo
        self.perceptual_weight = perceptual_weight
        # output log variance
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)
        # TODO 
        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                 n_layers=disc_num_layers,
                                                 use_actnorm=use_actnorm
                                                 ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        ##import ipdb; ipdb.set_trace()
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0] # NOTE paper equation (7)! nll_loss is from rec_loss (reconstruction loss): Delta_{G_L}[L_rec] and L_rec = rec_loss, G_L=last layer
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0] # Delta_{G_L}[L_GAN], generator_loss with four modules...
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4) # NOTE the real equation (7) d_weight=lambda
        # the reason behind equation (7): 如果rec_loss的对last_layer的梯度很大，而g_loss对于last_layer的梯度小，那么，lambda相对比较大，从而赋予g_loss更大的weight，如此，我们是希望model可以相对平衡地从两个loss上学习，而不是一个loss占据主导地位!!!
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self, inputs, reconstructions, posteriors, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train",
                weights=None):
        #import ipdb; ipdb.set_trace()
        # reconstruction loss = rec_loss, 重建loss
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous()) # inputs.shape=[1, 3, 256, 256]

        if self.perceptual_weight > 0:
            # 不是在原始的[1, 3, 256, 256] 空间比较了，而是修改了一下，是在vgg变换之后，的隐空间，进行对比
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss # NOTE a combination of two types of losses!

        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar # self.logvar = 0
        weighted_nll_loss = nll_loss
        if weights is not None:
            weighted_nll_loss = weights*nll_loss
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        kl_loss = posteriors.kl() # NOTE important, KL for a gaussian distribution!
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous()) # reconstructions = fake images; logits_fake.shape=[1, 1, 30, 30]
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
            g_loss = -torch.mean(logits_fake) # 取负值！want g_loss to be small -> logits_fake to be large -> the discriminator give high confidences for the 'fake images'

            if self.disc_factor > 0.0:
                try:
                    d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
                except RuntimeError:
                    assert not self.training
                    d_weight = torch.tensor(0.0)
            else:
                d_weight = torch.tensor(0.0)

            #import ipdb; ipdb.set_trace()
            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            # threshold=50001, 目的：前50000个iteration，都是disc_factor=0，理由：希望先好好地集中训练autoencoder，并且ignore gan loss，稳定之后，开始慢慢地开始gan的loss下的train！

            loss = weighted_nll_loss + self.kl_weight * kl_loss + d_weight * disc_factor * g_loss
            # when disc_factor=0, the third part of this loss will be ignored (gan-loss)
            # 即，前50000 个batch下，只用rec_loss和kl_loss!
            log = {"{}/total_loss".format(split): loss.clone().detach().mean(), 
                   "{}/logvar".format(split): self.logvar.detach(),
                   "{}/kl_loss".format(split): kl_loss.detach().mean(), 
                   "{}/nll_loss".format(split): nll_loss.detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   "{}/d_weight".format(split): d_weight.detach(),
                   "{}/disc_factor".format(split): torch.tensor(disc_factor),
                   "{}/g_loss".format(split): g_loss.detach().mean(),
                   }
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            if cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)
            # disc_factor=0 for the first 50000 iterations! NOTE 整体是，我们在利用discriminator来train autoencoder!
            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()
                   }
            return d_loss, log

