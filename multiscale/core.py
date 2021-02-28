
import torch
import torch.nn as nn
import torch.distributions as tdist
import torch.nn.functional as F

from multiscale.base import delayed_fm, ConvNormAct


class MultiScaleMergeBlock(nn.Module):
    def __init__(self,
                 layer_num,
                 base_fm,
                 fm_delay,
                 act,
                 norm,
                 scale_side: list,
                 scale_main: int = 1,
                 base_main: int = 512):  # main encoder input size
        super().__init__()
        base_main = int(base_main * ((1/2)**(layer_num+1)))  # spatial size of target after curr level's encoder block
        num_arms = len(scale_side) + 1
        self.scale_factor = [ss // scale_main for ss in scale_side]
        self.first_crop = [max(1, (base_main // ss) // 2) for ss in scale_side]
        self.second_crop = [0,]*(num_arms - 1)
        for layer, ss in enumerate(scale_side):
            if base_main <= ss:  # two-step merge
                self.second_crop[layer] = int(base_main // 2)
        in_channels = base_fm * (2 ** delayed_fm(layer_num+1, fm_delay))
        self.reduce_fm_size_to_original = ConvNormAct(in_channels=num_arms*in_channels,
                                                      out_channels=in_channels,
                                                      kernel_size=1,
                                                      stride=1,
                                                      act=act,
                                                      norm=norm)
        # set initial weights
        self.reduce_fm_size_to_original.module[0].weight.data.fill_(0)
        self.reduce_fm_size_to_original.module[0].weight.data[0:in_channels, 0:in_channels, 0, 0] = torch.eye(in_channels)
        self.reduce_fm_size_to_original.module[0].bias.data.fill_(0)
        # add jitter
        epsilon = 1e-4
        normaldist = tdist.Normal(torch.tensor([0.0]), torch.tensor([epsilon]))
        jitter_weights = normaldist.sample(self.reduce_fm_size_to_original.module[0].weight.data.shape)[..., 0]  # ignore dimension added at end
        jitter_biases = normaldist.sample(self.reduce_fm_size_to_original.module[0].bias.data.shape)[..., 0]
        self.reduce_fm_size_to_original.module[0].weight.data += jitter_weights
        self.reduce_fm_size_to_original.module[0].bias.data += jitter_biases

    def forward(self, *inputs, reduce):
        x_main = inputs[0]
        sides = list(inputs[1:])

        for x_side, first_crop, second_crop, scale_factor in zip(sides,
                                                                 self.first_crop,
                                                                 self.second_crop,
                                                                 self.scale_factor):
            # take central 4-bar
            half = [dim//2 for dim in x_side.shape]
            central_bar = [torch.tensor(range(h_-first_crop, h_+first_crop), device='cuda') for h_ in half]
            x_side = torch.index_select(x_side, dim=-1, index=central_bar[-1])
            x_side = torch.index_select(x_side, dim=-2, index=central_bar[-2])
            x_side = F.interpolate(x_side, scale_factor=scale_factor, mode="bilinear")

            if second_crop:
                # take central 16-bar (down to original size)
                half = [dim//2 for dim in x_side.shape]
                central_bar = [torch.tensor(range(h_-second_crop, h_+second_crop), device='cuda') for h_ in half]
                x_side = torch.index_select(x_side, dim=-1, index=central_bar[-1])
                x_side = torch.index_select(x_side, dim=-2, index=central_bar[-2]) # B x N x 16 x 16

            # merge paths
            x_main = torch.cat((x_main, x_side), dim=1)  # accumulates main + all sides
        if reduce:
            x_main = self.reduce_fm_size_to_original(x_main)
        return x_main


class LeakyGateBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
               #  act):
        super().__init__()
        self.merge = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0),
          #  act(inplace=True)
        )

    def forward(self, x, x_gated):
        x = torch.cat((x, x_gated), dim=1)
        x = self.merge(x)
        return x
