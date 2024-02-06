"""
    UpstreamExpert of MelHuBERT.
    Author: Tzu-Quan Lin (https://github.com/nervjack2)
    Reference: (https://github.com/s3prl/s3prl/tree/master/s3prl/upstream/distiller)
    Reference author: Heng-Jui Chang (https://github.com/vectominist)
"""

import yaml
import torch
import torchaudio
import numpy as np
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from ..interfaces import UpstreamBase
from .model import Wav2Vec2Model, Wav2Vec2Config

class UpstreamExpert(UpstreamBase):
    """
    The Wav2vec2 wrapper
    """
    def __init__(self, ckpt, mode, **kwargs):
        super().__init__(**kwargs)
        self.mode = mode 
        self.apply_padding_mask = True
        # Load upstream model 
        all_states = torch.load(ckpt, map_location="cpu")
        if "melhubert" in all_states["Upstream_Config"]:
            upstream_config = all_states["Upstream_Config"]["melhubert"] 
        else:
            upstream_config = all_states["Upstream_Config"]["model"] 
        upstream_config = Wav2Vec2Config(upstream_config)
        self.upstream_model = Wav2Vec2Model(upstream_config)
        state_dict = all_states["model"]

        # if self.mode == 'melhubert' or self.mode == 'distillation' or self.mode == 'row-pruning':
        #     self.upstream_model.load_state_dict(state_dict)
    
        if 'Pruned_heads' in all_states:
            # If head-pruned
            pruned_heads = all_states["Pruned_heads"]
            summarized = {}
            for layer_heads in pruned_heads:
                for layer in layer_heads:
                    summarized[layer] = summarized.get(layer, 0) + len(layer_heads[layer])
            pruned_heads = summarized

            for idx, layer in enumerate(self.upstream_model.encoder.layers):
                if idx in pruned_heads:
                    layer.self_attn.num_heads -= pruned_heads[idx]
                    orig_embed_dim = layer.self_attn.embed_dim
                    embed_dim = layer.self_attn.head_dim * layer.self_attn.num_heads
                    bias = True
                    layer.self_attn.embed_dim = embed_dim
                    layer.self_attn.k_proj = nn.Linear(orig_embed_dim, embed_dim, bias=bias)
                    layer.self_attn.v_proj = nn.Linear(orig_embed_dim, embed_dim, bias=bias)
                    layer.self_attn.q_proj = nn.Linear(orig_embed_dim, embed_dim, bias=bias)
                    layer.self_attn.out_proj = nn.Linear(embed_dim, orig_embed_dim, bias=bias)
                    layer.self_attn.skip_embed_dim_check = True
                    layer.self_attn.reset_parameters()   

        if 'Pruning' in all_states:
            # If weight-pruned
            from ..s3prl_upstream.pytorch_code import prune
            from ..s3prl_upstream.wp_utils import get_params_to_prune
            params_to_prune, _ = get_params_to_prune(self.upstream_model)
            prune.global_unstructured(
                params_to_prune,
                pruning_method=prune.Identity,
            )
        
        self.upstream_model.load_state_dict(state_dict)
        
        if 'Pruning' in all_states:
            for module, name in params_to_prune:
                prune.remove(module, name)
        
    def get_downsample_rates(self, key: str) -> int:
        return 320
    
    def zero_mean_unit_var_norm(input_values):
        """
        Every array in the list is normalized to have zero mean and unit variance
        Taken from huggingface to ensure the same behavior across s3prl and huggingface
        Reference: https://github.com/huggingface/transformers/blob/a26f4d620874b32d898a5b712006a4c856d07de1/src/transformers/models/wav2vec2/feature_extraction_wav2vec2.py#L81-L86
        """
        return [(x - np.mean(x)) / np.sqrt(np.var(x) + 1e-5) for x in input_values]

    def forward(self, wavs):
        device = wavs[0].device
        # if self.wav_normalize:
        #     if self.numpy_wav_normalize:
        #         wavs = self.zero_mean_unit_var_norm([wav.cpu().numpy() for wav in wavs])
        #         wavs = [torch.from_numpy(wav).to(device) for wav in wavs]
        #     else:
        #         wavs = [F.layer_norm(wav, wav.shape) for wav in wavs]

        wav_lengths = torch.LongTensor([len(wav) for wav in wavs]).to(device)
        wav_padding_mask = ~torch.lt(
            torch.arange(max(wav_lengths)).unsqueeze(0).to(device),
            wav_lengths.unsqueeze(1),
        )
        padded_wav = pad_sequence(wavs, batch_first=True)

        results = self.upstream_model.extract_features(
            padded_wav, wav_padding_mask if self.apply_padding_mask else None
        )
        
        return {
            "hidden_states": [
                h[0].transpose(0, 1) for h in results["layer_results"]
            ],
        }

