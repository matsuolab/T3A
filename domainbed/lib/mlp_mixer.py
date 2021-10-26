import torch
import timm
from domainbed.lib.vision_transformer import Identity


class MLPMixer(torch.nn.Module):
    KNOWN_MODELS = {
        'Mixer-B16': timm.models.mlp_mixer.mixer_b16_224_in21k,
        'Mixer-L16': timm.models.mlp_mixer.mixer_l16_224_in21k,
    }
    def __init__(self, input_shape, hparams):
        super().__init__()
        func = self.KNOWN_MODELS[hparams['backbone']]
        self.network = func(pretrained=True)
        self.n_outputs = self.network.norm.normalized_shape[0]
        self.network.head = Identity()
        self.hparams = hparams

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        return self.network(x)
