
import torch
from multiscale.models.Ymm2classRes18Net import Ymm2classRes18Net

num_channels = 3
batch_size = 6  # Note: only for the sake of this deno -- should usually be bigger for batch norm

# suppose we are given concentric image patches for the spatial scales 1 (e.g. 0.2 um/px), 4
# (e.g. 0.8 um/px) and 16 (1.6 um/px), each sized 512x512 pixels
image_patch_scale1 = torch.rand(size=[batch_size, num_channels, 512, 512], device='cuda')
image_patch_scale4 = torch.rand(size=[batch_size, num_channels, 572, 572], device='cuda')
image_patch_scale16 = torch.rand(size=[batch_size, num_channels, 512, 512], device='cuda')

# assembled into a sequence with one element per scale, each of the size B x C x W x H
# (Note: width and height do *not* have to be the same per scale, but should be compatible with the downsampling steps
# inside the encoders)
image_patch = tuple([
    image_patch_scale1,
    image_patch_scale4,
    image_patch_scale16
])

# and a basic configurate might, such as, for instance
cfg = {
    'num_classes': 4,
    'num_channels': num_channels,
    'activation_function': 'ReLU',
    'num_base_featuremaps': 64,
    'encoder_featuremap_delay': 2,
    'decoder_featuremaps_out': [512, 256, 256, 128, -1],
    'conv_norm_type': 'None',
    'depth_levels_down_main': [[2, 3], [0, 3], [0, 1], [0, 1], [0, 1]],  # [Convs,Res] each
    'depth_levels_down_side': [[2, 3], [0, 3], [0, 1], [0, 1], [0, 1]],  # [Convs,Res] each
    'depth_levels_down_tail': [[2, 3], [0, 3], [0, 1], [0, 1], [0, 1]],  # [Convs,Res] each
    'depth_levels_up': [1, 1, 1, 1, 1],  # Convs
    'depth_bottleneck': [0, 0, 0],  # [Conv,Res,Conv]
    'internal_prediction_activation': 'None',  # Softmax, Sigmoid or None. None for use with BCEWithLogitsLoss etc.
}

# we can then instantiate and apply an examplary multi-scale model using the syntax
model = Ymm2classRes18Net(cfg=cfg).to(device='cuda')

# where the return values are segmentation_logits and classification_logits. Both can of course be used to compote a
# training loss for our example model
segmentation_logits, classification_logits = model(*image_patch)

# done
print('Demo passed. Many thanks for your interest in our work!\n\n'
      'Have fun using/ playing around with/ extending/ improving/ beating our architectures\n'
      '-- and do not hesitate to contact us!\n\n\t\t\t\t\t\t\t\t\t\t\t\tRuediger Schmitz / r.schmitz AT uke.de')
