import torch
import torch.nn as nn
from torchvision import models
from collections import OrderedDict

from multiscale.base import BaseNet, ResBottleneckBlock, ResEncoderBlock, SkipBlock, DecoderBlock
from multiscale.core import MultiScaleMergeBlock


class YclassRes18Net(BaseNet):
    """
        - ResNet18 backbone
        - Y architecture: spatials scales x1 and x16
        - classification output for deep guidance of x16-encoder
    """
    def __init__(self, cfg):
        super().__init__(cfg=cfg)

        self.depth_levels_tail_encoder = cfg['depth_levels_down_tail']

        base_model = models.resnet18(pretrained=True)
        base_layers = list(base_model.children())
        main_base = {0: base_layers[:3],
                     1: base_layers[3:5],
                     2: [base_layers[5]],
                     3: [base_layers[6]],
                     4: [base_layers[7]]}

        base_model_tail = models.resnet18(pretrained=True)
        num_ftrs = base_model_tail.fc.in_features
        if num_ftrs is not self.num_classes:
            base_model_tail.fc = nn.Sequential(OrderedDict([
                ('fc1', nn.Linear(num_ftrs, self.num_classes))
            ]))
        tail_layers = list(base_model_tail.children())
        tail_base = {0: tail_layers[:3],
                     1: tail_layers[3:5],
                     2: [tail_layers[5]],
                     3: [tail_layers[6]],
                     4: [tail_layers[7]]}
        tail_remainder = tail_layers[8:]

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

        self.conv_before_out = nn.Conv2d(in_channels=self.base_fm,
                                         out_channels=self.num_classes,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0)

        self.pred_act = self.get_pred_act(self.pred_act_type)

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
                          norm=self.get_norm(self.norm_type)
                )
            )
            self.decoder.append(
                DecoderBlock(act=self.get_act(self.act_type),
                             norm=self.get_norm(self.norm_type),
                             decoder_fm_out=self.decoder_fm_out,  # target fm sizes for each decoder level
                             base_fm=self.base_fm,
                             fm_delay=self.fm_delay,
                             layer_depth=self.depth_levels_decoder[layer],
                             layer_num=layer
                )
            )
        # only one merge (at deepest level)
        self.merge = MultiScaleMergeBlock(
            layer_num=4,
            base_fm=self.base_fm,
            fm_delay=self.fm_delay,
            scale_side=[16],
            act=self.get_act(self.act_type),
            norm=self.get_norm(self.norm_type)
        )

        # build tail encoder
        self.tail_encoder = nn.ModuleList()
        for layer in range(len(tail_base) + 1):
            if layer < len(tail_base):
                self.tail_encoder.append(
                    ResEncoderBlock(
                        res_layers=tail_base[layer],
                        act=self.get_act(self.act_type),
                        norm=self.get_norm(self.norm_type),
                        base_fm=self.base_fm,
                        fm_delay=self.fm_delay,
                        layer_depth=self.depth_levels_tail_encoder[layer],
                        layer_num=layer,
                        in_channels=self.num_channels if layer == 0 else None,
                    )
                )
            else:
                self.tail_encoder.extend(
                    tail_remainder
                )

    def forward(self, x_main, x_tail):
        # main path
        from_skip = []
        for layer, (main, tail) in enumerate(zip(self.main_encoder,
                                                 self.tail_encoder
                                                )
                                            ):
            x_tail, _ = tail(x_tail)  # tail path (no skip conns)
            x_main, for_skip = main(x_main)
            from_skip.append(self.skip[layer](for_skip))

        x = self.merge(x_main, x_tail, reduce=False)

        # bottleneck
        x = self.bottleneck(x)

        for layer in reversed(range(self.deepest_layer_num)):
            x = self.decoder[layer](x, from_skip[layer])

        # rest of tail
        x_tail = self.tail_encoder[-2](x_tail)
        x_tail = self.tail_encoder[-1](torch.squeeze(x_tail))

        x = self.conv_before_out(x)

        # usually, pred_act = Identity, so logits from both main/side and tail are returned, not probas
        x = self.pred_act(x)
        x_tail = self.pred_act(x_tail)

        return x, x_tail
