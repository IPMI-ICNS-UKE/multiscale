
import torch
import torch.nn as nn


def delayed_fm(layer, fm_delay):
    return max(0, layer-fm_delay)


class ConvNormAct(nn.Module):
    def __init__(self,
                 in_channels,
                 act,
                 norm,
                 out_channels=None,
                 stride=1,
                 kernel_size=1):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        padding = int((kernel_size - 1) / 2)
        self.module = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding),
            norm(out_channels),
            act(inplace=True)
        )

    def forward(self, *input):
        return self.module(*input)


class DownBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 act,
                 norm,
                 kernel_size=1,
                 factor=2
                 ):
        super().__init__()
        out_channels = in_channels  # always
        padding = int((kernel_size - 1) / 2)
        self.module = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      stride=factor,
                      padding=padding),
            norm(out_channels),
            act(inplace=True)
        )

    def forward(self, *input):
        return self.module(*input)


class UpBlock(nn.Module):
    def __init__(self,
                 factor=2):
        super().__init__()
        self.up = nn.Upsample(scale_factor=factor, mode='bilinear', align_corners=True)

    def forward(self, *input):
        return self.up(*input)


class ResEncoderBlock(nn.Module):
    def __init__(self,
                 act,
                 norm,
                 base_fm,
                 layer_num,
                 res_layers,
                 fm_delay=0,
                 layer_depth=2,
                 in_channels=None,
                 kernel_size=3):
        super().__init__()
        if layer_num == 0 and in_channels is None:
            raise AssertionError('The first encoder block needs to know the number of input channels.')
        elif in_channels is None:
            in_channels = base_fm * (2 ** (delayed_fm(layer_num, fm_delay) - 1))
        out_channels = base_fm * (2 ** delayed_fm(layer_num, fm_delay))
        if layer_depth[0] > 0:
            temp = []
            # add first block (extending fm to target size)
            temp.append(
                ConvNormAct(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=kernel_size,
                          act=act,
                          norm=norm)
            )
            # add other blocks (leaving fm size untouched)
            for ii in range(1, layer_depth[0]):
                temp.append(
                    ConvNormAct(in_channels=out_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              act=act,
                              norm=norm
                    )
                )
            self.convs_before_skip = nn.Sequential(*temp)
        else:
            self.convs_before_skip = nn.Identity()
        self.res_and_down = nn.Sequential(*res_layers)

    def forward(self, x):
        for_skip = self.convs_before_skip(x)
        for_next_layer = self.res_and_down(x)
        return for_next_layer, for_skip


class ConvEncoderBlock(nn.Module):
    def __init__(self,
                 act,
                 norm,
                 base_fm,
                 layer_num,
                 fm_delay=0,
                 layer_depth=2,
                 in_channels=None,
                 kernel_size=3):
        super().__init__()
        if layer_num == 0 and in_channels is None:
            raise AssertionError('The first encoder block needs to know the number of input channels.')
        elif in_channels is None:
            in_channels = base_fm * (2 ** (delayed_fm(layer_num, fm_delay) - 1))
        out_channels = base_fm * (2 ** delayed_fm(layer_num, fm_delay))
        temp = []
        # add first block (extending fm to target size)
        temp.append(
            ConvNormAct(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      act=act,
                      norm=norm)
        )
        # add other blocks (leaving fm size untouched)
        for ii in range(1, layer_depth):
            temp.append(
                ConvNormAct(in_channels=out_channels,
                          out_channels=out_channels,
                          kernel_size=kernel_size,
                          act=act,
                          norm=norm
                )
            )
        self.horizontals = nn.Sequential(*temp)
        self.down = DownBlock(in_channels=out_channels,
                              act=act,
                              norm=norm)

    def forward(self, x):
        for_skip = self.horizontals(x)
        for_next_layer = self.down(for_skip)
        return for_next_layer, for_skip


class ResBottleneckBlock(nn.Module):
    def __init__(self,
                 act,
                 norm,
                 base_fm,
                 layer_num,
                 res_layers=None,
                 fm_delay=0,
                 layer_depth=[0,0,0],
                 kernel_size=3,
                 num_arms=1):
        super().__init__()
        in_channels = base_fm * (2 ** (delayed_fm(layer_num, fm_delay))) * num_arms
        out_channels = base_fm * (2 ** delayed_fm(layer_num, fm_delay))
        temp = []
        # add Convs before Res (default: off)
        if layer_depth[0] > 0:
            # add first block (extending fm to target size)
            temp.append(
                ConvNormAct(in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            act=act,
                            norm=norm)
            )
            # add other blocks (leaving fm size untouched)
            for ii in range(1, layer_depth[0]):
                temp.append(
                    ConvNormAct(in_channels=out_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                act=act,
                                norm=norm
                                )
                )
        # add Res
        if res_layers:
            temp.append(*res_layers)
        # add Convs after Res
        if layer_depth[2] > 0:
            # leaving fm size unchanged
            in_channels = out_channels if res_layers or layer_depth[0] > 0 else in_channels
            for ii in range(layer_depth[2]):
                if ii>0:
                    in_channels = out_channels
                temp.append(
                    ConvNormAct(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                act=act,
                                norm=norm
                                )
                )
        # add finishing 1x1 Conv
        in_channels = out_channels if res_layers or layer_depth[0] > 0 or layer_depth[2] > 0 else in_channels
        temp.append(
            ConvNormAct(in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=1,
                        stride=1,
                        act=act,
                        norm=norm)
        )
        self.horizontals = nn.Sequential(*temp)

    def forward(self, x):
        return self.horizontals(x)


class ConvBottleneckBlock(nn.Module):
    def __init__(self,
                 act,
                 norm,
                 base_fm,
                 layer_num,
                 fm_delay=0,
                 layer_depth=2,
                 kernel_size=3):
        super().__init__()
        in_channels = base_fm * (2 ** (delayed_fm(layer_num, fm_delay) - 1))
        out_channels = base_fm * (2 ** delayed_fm(layer_num, fm_delay))
        temp = []
        # add first block (extending fm to target size)
        temp.append(
            ConvNormAct(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      act=act,
                      norm=norm)
        )
        # add other blocks (leaving fm size untouched)
        for ii in range(1, layer_depth):
            temp.append(
                ConvNormAct(in_channels=out_channels,
                          out_channels=out_channels,
                          kernel_size=kernel_size,
                          act=act,
                          norm=norm
                )
            )
        self.horizontals = nn.Sequential(*temp)

    def forward(self, x):
        return self.horizontals(x)


class SkipBlock(nn.Module):
    def __init__(self,
                 base_fm,
                 layer_num,
                 act,
                 norm,
                 fm_delay=0):
        super().__init__()
        in_channels = base_fm * (2 ** delayed_fm(layer_num, fm_delay))
        out_channels = in_channels
        self.skip = ConvNormAct(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=1,
                                stride=1,
                                act=act,
                                norm=norm)

    def forward(self, x):
        return self.skip(x)


class CombineBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, from_below, from_skip):
        return torch.cat((from_below, from_skip), dim=1)


class DecoderBlock(nn.Module):
    def __init__(self,
                 act,
                 norm,
                 base_fm,
                 layer_num,
                 decoder_fm_out=None,
                 fm_delay=0,
                 layer_depth=2,
                 kernel_size=3
                 ):
        super().__init__()
        in_channels_from_skip = base_fm * (2 ** delayed_fm(layer_num, fm_delay))
        if decoder_fm_out is None:
            in_channels_from_below = base_fm * (2 ** (delayed_fm(layer_num, fm_delay) + 1))
            out_channels = base_fm * (2 ** delayed_fm(layer_num, fm_delay))
        else:
            try:
                in_channels_from_below = decoder_fm_out[layer_num+1]  # one layer deeper
            except IndexError:
                in_channels_from_below = base_fm * (2 ** (delayed_fm(layer_num, fm_delay) + 1))
            out_channels = decoder_fm_out[layer_num]  # this layer
            if out_channels == -1:
                out_channels = base_fm
        self.up = UpBlock()
        self.combine = CombineBlock()
        temp = []
        # add first block (reducing fm to target size)
        temp.append(
            ConvNormAct(in_channels=in_channels_from_skip+in_channels_from_below,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      act=act,
                      norm=norm)
        )
        # add other blocks (leaving fm size untouched)
        for ii in range(1, layer_depth):
            temp.append(
                ConvNormAct(in_channels=out_channels,
                          out_channels=out_channels,
                          kernel_size=kernel_size,
                          act=act,
                          norm=norm
                )
            )
        self.horizontals = nn.Sequential(*temp)

    def forward(self, from_below, from_skip):
        x = self.up(from_below)
        x = self.combine(x, from_skip)
        x = self.horizontals(x)
        return x


class OutBlock(nn.Module):
    def __init__(self,
                 base_fm,
                 num_classes,
                 pred_act):
        super().__init__()
        self.out = nn.Sequential(
            nn.Conv2d(in_channels=base_fm,
                      out_channels=num_classes,
                      kernel_size=1,
                      stride=1),
            pred_act
        )

    def forward(self, x):
        return self.out(x)


class BaseNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_classes = cfg['num_classes']
        self.num_channels = cfg['num_channels']
        self.fm_delay = cfg.get('encoder_featuremap_delay', 0)
        self.decoder_fm_out = cfg.get('decoder_featuremaps_out', None)
        if self.decoder_fm_out is not None:
            self.decoder_fm_out = list(reversed(self.decoder_fm_out))

        self.base_fm = cfg['num_base_featuremaps']
        self.norm_type = cfg['conv_norm_type']
        self.act_type = cfg['activation_function']
        self.pred_act_type = cfg['internal_prediction_activation']

        self.depth_levels_main_encoder = cfg['depth_levels_down_main']

        self.depth = len(self.depth_levels_main_encoder)
        self.depth_levels_decoder = list(reversed(cfg.get('depth_levels_up', [2, ] * (self.depth))))
        self.deepest_layer_num = len(self.depth_levels_decoder)
        self.depth_bottleneck = cfg['depth_bottleneck']
        assert len(self.depth_levels_main_encoder) == len(self.depth_levels_decoder), "Encoder and decoder need to " \
                                                                                 "have the same depth"

        # check base model requirement
        assert self.fm_delay == 2, "encoder_featuremap_delay must be 2 for ResNet18-based UNet"
        assert self.depth_levels_main_encoder[0] == [2,3] \
               and self.depth_levels_main_encoder[1] == [0,3] \
               and self.depth_levels_main_encoder[2] == [0,1] \
               and self.depth_levels_main_encoder[3] == [0,1]\
               and self.depth_levels_main_encoder[4] == [0,1], "depth_levels_downs needs to be" \
                                                          " [[2,3],[0,3],[0,1],[0,1],[0,1]]" \
                                                          " ResNet18-based UNet"
        assert self.depth_bottleneck == [0, 0, 0], "depth_bottleneck needs to be [0,0,0] for ResNet18-based UNet"
        assert self.decoder_fm_out == list(reversed([512, 256, 256, 128, -1])), "decoder_featuremaps_out needs to be" \
                                                           "[512, 256, 256, 128, -1] for ReNet18-based UNet"

    @staticmethod
    def get_act(act_type):
        if act_type == 'ReLU':
            act = nn.ReLU#
        elif act_type == 'LeakyReLU':
            act = nn.LeakyReLU
        elif act_type == 'SeLU':
            act = nn.SELU
        elif act_type == 'Sigmoid':
            act = nn.Sigmoid
        elif act_type == 'Tanh':
            act = nn.Tanh
        else:
            raise NotImplementedError
        return act

    @staticmethod
    def get_norm(norm_type):
        if norm_type == 'batch_norm':
            norm = nn.BatchNorm2d
        elif norm_type == 'instance_norm':
            norm = nn.InstanceNorm2d
        elif norm_type == 'None':
            norm = nn.Identity
        else:
            raise NotImplementedError
        return norm

    @staticmethod
    def get_pred_act(pred_act_type):
        if pred_act_type == 'Sigmoid':
            pred_act = nn.Sigmoid()
        elif pred_act_type == 'Softmax':
            pred_act = nn.Softmax(dim=1)
        elif pred_act_type == 'None':
            pred_act = nn.Identity()
        else:
            raise NotImplementedError
        return pred_act

