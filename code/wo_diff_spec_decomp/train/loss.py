import torch
import torch.nn as nn
from torch import autograd
import torchvision.models as models
import numpy as np


Tensor = torch.cuda.FloatTensor


class GradientPenaltyLoss(nn.Module):
    def __init__(self, device):
        super(GradientPenaltyLoss, self).__init__()
        self.device = device
        self.register_buffer('grad_outputs', torch.Tensor())
        self.grad_outputs = self.grad_outputs.to(device)

    def get_grad_outputs(self, input):
        if self.grad_outputs.size() != input.size():
            self.grad_outputs.resize_(input.size()).fill_(1.0)
        return self.grad_outputs

    def forward(self, D, real_data, fake_data):
        # random weight term for interpolation between real and fake samples
        alpha = Tensor(real_data.shape[0], 1, 1, 1).to(self.device)
        alpha.uniform_()
        # get random interpolation between real and fake samples
        interpolates = alpha * fake_data.detach() + (1 - alpha) * real_data
        interpolates.requires_grad = True
        pred_d_interpolates = D(interpolates)

        grad_outputs = self.get_grad_outputs(pred_d_interpolates)
        grad_interp = torch.autograd.grad(outputs=pred_d_interpolates, inputs=interpolates, grad_outputs=grad_outputs,
                                          create_graph=True, retain_graph=True, only_inputs=True)[0]
        grad_interp = grad_interp.reshape((grad_interp.size(0), -1))
        grad_interp_norm = grad_interp.norm(2, dim=1)

        loss = ((grad_interp_norm - 1) ** 2).mean()
        return loss


# Paper: Wasserstein Divergence for GANs
# gradient penalty term in Wasserstein Divergence
class WDivGradientPenaltyLoss(nn.Module):
    def __init__(self):
        super(WDivGradientPenaltyLoss, self).__init__()

    def forward(self, discriminator, real_data, fake_data, p=6):
        # random weight term for interpolation between real and fake samples
        alpha = Tensor(np.random.random((real_data.shape[0], 1, 1, 1)))
        # get random interpolation between real and fake samples
        interpolates = (alpha * real_data + ((1 - alpha) * fake_data)).requires_grad_(True)
        pred_d_interpolates = discriminator(interpolates)
        fake_grad_outputs = autograd.Variable(Tensor(real_data.shape[0], 1).fill_(1.0), requires_grad=False)
        gradients = autograd.grad(outputs=pred_d_interpolates, inputs=interpolates, grad_outputs=fake_grad_outputs,
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.contiguous().view(gradients.shape[0], -1)
        gradient_penalty = (gradients.pow(2).sum(1) ** (p / 2)).mean()
        return gradient_penalty


class GANLoss(nn.Module):
    def __init__(self, type='nsgan', target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.type = type
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        if type == 'nsgan':
            self.criterion = nn.BCELoss()
        elif type == 'wgan':
            self.criterion = self.wgan_loss
        elif type == 'lsgan':
            self.criterion = nn.MSELoss()
        elif type == 'hinge':
            self.criterion = self.hinge_loss
        else:
            raise NotImplementedError("GAN type %s is not found!" % type)

    def wgan_loss(self, input, target):
        return -1 * input.mean() if target else input.mean()

    def hinge_loss(self, input, target, is_discriminator):
        criterion = nn.ReLU()
        if is_discriminator:
            return criterion(1 - input).mean() if target else criterion(1 + input).mean()
        else:
            return (-input).mean()

    def get_target_tensor(self, input, target_is_real):
        if self.type == 'wgan':
            return target_is_real

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def forward(self, input, target_is_real, is_discriminator=None):
        if self.type == 'hinge':
            loss = self.criterion(input, target_is_real, is_discriminator)
        else:
            target_tensor = self.get_target_tensor(input, target_is_real)
            loss = self.criterion(input, target_tensor)
        return loss


class L1ReconstructionLoss(nn.Module):
    def __init__(self):
        super(L1ReconstructionLoss, self).__init__()
        self.l1_loss = torch.nn.L1Loss()

    def forward(self, input, target):
        loss = self.l1_loss(input, target)
        return loss


class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bce_loss = torch.nn.BCELoss()

    def forward(self, input, target):
        loss = self.bce_loss(input, target)
        return loss


class BCELossLogits(nn.Module):
    def __init__(self):
        super(BCELossLogits, self).__init__()
        self.bce_loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, input, target):
        loss = self.bce_loss(input, target)
        return loss
    

class ToneMappingLoss(nn.Module):
    def __init__(self):
        super(ToneMappingLoss, self).__init__()
        self.l1_loss = L1ReconstructionLoss()

    def forward(self, input, target):
        return self.l1_loss(input/(input+1), target/(target+1))


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg16 = models.vgg16(pretrained=True).cuda()
        self.vgg_layers = vgg16.features
        self.l1_loss = L1ReconstructionLoss()
        self.layer_name_mapping = {
            '4': "pool_1",
            '9': "pool_2",
            '16': "pool_3",
        }

    def forward(self, input, target):
        loss = 0
        for layer, module in self.vgg_layers._modules.items():
            input = module(input)
            target = module(target)
            if layer in self.layer_name_mapping:
                loss += self.l1_loss(input, target)
        return loss



