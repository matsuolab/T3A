import torch
import timm


class Identity(torch.nn.Module):
    """An identity layer"""
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

""" 
Note: The implementation is not used main experimennts. Please use ViT2. 

class ViT(torch.nn.Module):
    def __init__(self, input_shape, hparams):
        super().__init__()
        self.network = pytorch_pretrained_vit.ViT(
            hparams['backbone'], pretrained=True
        )
        self.n_outputs = self.network.fc.in_features
        del self.network.fc
        self.network.fc = Identity()
        self.hparams = hparams

    def forward(self, x):
        return self.network(x)
"""


class ViT2(torch.nn.Module):
    KNOWN_MODELS = {
        'ViT-B16': timm.models.vision_transformer.vit_base_patch16_224_in21k, 
        'ViT-B32': timm.models.vision_transformer.vit_base_patch32_224_in21k,
        'ViT-L16': timm.models.vision_transformer.vit_large_patch16_224_in21k,
        'ViT-L32': timm.models.vision_transformer.vit_large_patch32_224_in21k,
        'ViT-H14': timm.models.vision_transformer.vit_huge_patch14_224_in21k
    }

    def __init__(self, input_shape, hparams):
        super().__init__()
        func = self.KNOWN_MODELS[hparams['backbone']]
        self.network = func(pretrained=True)
        self.n_outputs = self.network.norm.normalized_shape[0]
        self.network.head = Identity()
        if hasattr(self.network, 'head_dist'):
            self.network.head_dist = None
        self.hparams = hparams

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        return self.network(x)

    def forward_adaptive_token(self, x, cls_token):
        x = self.network.patch_embed(x)
        if cls_token.shape[0] == 1:
            cls_token = cls_token.expand(x.shape[0], -1, -1)
        
        if self.network.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.network.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.network.pos_drop(x + self.network.pos_embed)
        x = self.network.blocks(x)
        x = self.network.norm(x)
        if self.network.dist_token is None:
            return self.network.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]        


class DINO(torch.nn.Module):
    """ 
    Note: The implementation is not used in main experimennts. 
    """
    def __init__(self, input_shape, hparams):
        super().__init__()
        self.network = torch.hub.load('facebookresearch/dino:main', hparams['backbone'])
        self.n_outputs = self.network.norm.normalized_shape[0]
        self.hparams = hparams

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        return self.network(x)


class DeiT(torch.nn.Module):
    KNOWN_MODELS = {
        'DeiT': timm.models.vision_transformer.vit_deit_base_distilled_patch16_224
    }
    def __init__(self, input_shape, hparams):
        super().__init__()
        func = self.KNOWN_MODELS[hparams['backbone']]
        self.network = func(pretrained=True)
        self.n_outputs = self.network.norm.normalized_shape[0]
        self.network.head = Identity()
        if hasattr(self.network, 'head_dist'):
            self.network.head_dist = None
        self.hparams = hparams

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        y = self.network(x)
        return (y[0] + y[1]) / 2  # This is the default option during inference of DeiT k


class HybridViT(torch.nn.Module):
    KNOWN_MODELS = {
        'HViT': timm.models.vision_transformer_hybrid.vit_base_r50_s16_224_in21k
    }
    def __init__(self, input_shape, hparams):
        super().__init__()
        func = self.KNOWN_MODELS[hparams['backbone']]
        self.network = func(pretrained=True)
        self.n_outputs = self.network.norm.normalized_shape[0]
        self.network.head = Identity()
        if hasattr(self.network, 'head_dist'):
            self.network.head_dist = None
        self.hparams = hparams

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        return self.network(x)
