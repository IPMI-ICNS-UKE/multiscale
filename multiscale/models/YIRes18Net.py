import torch
import torch.nn as nn
from torchvision import models
from collections import OrderedDict
import torch.nn.functional as F

from multiscale.base import BaseNet, ResBottleneckBlock, ResEncoderBlock, SkipBlock, DecoderBlock
from multiscale.core import MultiScaleMergeBlock, LeakyGateBlock


class Res18c4YleakyI18Net(BaseNet):
    """
        - ResNet18 backbone
        - YI architecture: spatials scales x1 and x4 for Y, plus x16-encoder to context-gate
        - classification output for deep guidance of x16-encoder
    """
    def __init__(self, cfg):
        super().__init__(cfg=cfg)

        self.depth_levels_side_encoder = cfg['depth_levels_down_side']

        base_model = models.resnet18(pretrained=True)
        base_layers = list(base_model.children())
        main_base = {0: base_layers[:3],
                     1: base_layers[3:5],
                     2: [base_layers[5]],
                     3: [base_layers[6]],
                     4: [base_layers[7]]}

        base_model_side = models.resnet18(pretrained=True) # deepcopy(base_model)
        side_layers = list(base_model_side.children())
        side_base = {0: side_layers[:3],
                     1: side_layers[3:5],
                     2: [side_layers[5]],
                     3: [side_layers[6]],
                     4: [side_layers[7]]}

        base_model_tail = models.resnet18(pretrained=True)
        num_ftrs = base_model_tail.fc.in_features
        if num_ftrs is not self.num_classes:
            base_model_tail.fc = nn.Sequential(OrderedDict([
                ('fc1', nn.Linear(num_ftrs, self.num_classes))
            ]))

        self.bottleneck = ResBottleneckBlock(
            res_layers=None,
            act=self.get_act(self.act_type),
            norm=self.get_norm(self.norm_type),
            base_fm=self.base_fm,
            fm_delay=self.fm_delay,
            layer_num=self.deepest_layer_num,
            layer_depth=self.depth_bottleneck,
            num_arms=2
        )

        self.leak = LeakyGateBlock(
            in_channels=2*self.num_classes,
            out_channels=self.num_classes
        )

        self.conv_before_out = nn.Conv2d(in_channels=self.base_fm,
                                         out_channels=self.num_classes,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0)

        self.pred_act=self.get_pred_act(self.pred_act_type)

        # build tail classifier
        self.tail = base_model_tail

        # build deocder + main encoder
        self.main_encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.skip = nn.ModuleList()
        for layer in range(len(self.depth_levels_main_encoder)):
            self.main_encoder.append(
                ResEncoderBlock(
                    res_layers=main_base[layer],
                    act=self.get_act(self.act_type),
                    norm=self.get_norm(self.norm_type),
                    base_fm=self.base_fm,
                    fm_delay=self.fm_delay,
                    layer_depth=self.depth_levels_main_encoder[layer],
                    layer_num=layer,
                    in_channels=self.num_channels if layer == 0 else None
                )
            )
            self.skip.append(
                SkipBlock(base_fm=self.base_fm,
                          fm_delay=self.fm_delay,
                          layer_num=layer,
                          act=self.get_act(self.act_type),
                          norm=self.get_norm(self.norm_type))
            )
            self.decoder.append(
                DecoderBlock(act=self.get_act(self.act_type),
                             norm=self.get_norm(self.norm_type),
                             decoder_fm_out=self.decoder_fm_out,  # target fm sizes for each decoder level
                             base_fm=self.base_fm,
                             fm_delay=self.fm_delay,
                             layer_depth=self.depth_levels_decoder[layer],
                             layer_num=layer)
            )

        # build side encoder
        self.side_encoder = nn.ModuleList()
        for layer in range(len(self.depth_levels_side_encoder)):
            self.side_encoder.append(
                ResEncoderBlock(
                    res_layers=side_base[layer],
                    act=self.get_act(self.act_type),
                    norm=self.get_norm(self.norm_type),
                    base_fm=self.base_fm,
                    fm_delay=self.fm_delay,
                    layer_depth=self.depth_levels_side_encoder[layer],
                    layer_num=layer,
                    in_channels=self.num_channels if layer == 0 else None,
                )
            )

    def forward(self, x_main, x_side, x_tail):
        # main path
        from_skip = []
        for layer, encode in enumerate(self.main_encoder):
            for_next_layer, for_skip = encode(x_main)
            x_main = for_next_layer
            from_skip.append(self.skip[layer](for_skip))

        #side path (no skip conns)
        for layer, encode in enumerate(self.side_encoder):
            x_side, _ = encode(x_side)

        # take central 4-bar
        half = [dim//2 for dim in x_side.shape]
        central_bar = [torch.tensor(range(h_-2, h_+2), device='cuda') for h_ in half]
        x_side = torch.index_select(x_side, dim=-1, index=central_bar[-1])
        x_side = torch.index_select(x_side, dim=-2, index=central_bar[-2])
        x_side = F.interpolate(x_side, scale_factor=4, mode="bilinear")

        # merge paths
        x = torch.cat((x_main, x_side), dim=1)

        # bottleneck
        x = self.bottleneck(x)

        for layer in reversed(range(self.deepest_layer_num)):
            x = self.decoder[layer](x, from_skip[layer])

        # tail classifier
        x_tail = self.tail.forward(x_tail)
        x_tail_classifications = torch.sigmoid(x_tail)

        x = self.conv_before_out(x)
        x_gated = torch.einsum('bc,bcij->bcij', x_tail_classifications, x)
        # leaky
        x = self.leak(x, x_gated)

        # usually, pred_act = Identity, so logits from both main/side and tail are returned, not probas
        x = self.pred_act(x)
        x_tail = self.pred_act(x_tail)

        return x, x_tail
