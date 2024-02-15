"""
    Training interface of MelHuBERT.
    Author: Tzu-Quan Lin (https://github.com/nervjack2)
"""
# Just copy from melhubert pretraining, haven't been modified yet.
# Need to align with config
import yaml

import torch
import torch.nn as nn
from model import Wav2Vec2Model, Wav2Vec2Config
from weight_pruning.wp_utils import get_params_to_prune
from criterion import Wav2vecCriterion
from pytorch_code import prune

class Wav2vec2Pretrainer(nn.Module):
    def __init__(self, upstream_config, initial_weight=None, device='cuda', multi_gpu=False, **kwargs):
        super(Wav2vec2Pretrainer, self).__init__()

        self.initial_weight = initial_weight
        self.device = device
        self.multi_gpu = multi_gpu

        self.upstream_config = upstream_config
        # Initialize the model 
        self._init_model()
        # Define pre-training loss
        self.criterion = Wav2vecCriterion()
     
        # Make multiple GPU training possible
        if self.multi_gpu:
            self.model = torch.nn.DataParallel(self.model)
            print('[Pretrainer] - Multi-GPU training Enabled: ' + str(torch.cuda.device_count()))
        print('[Pretrainer] - Number of parameters: ' + str(sum(p.numel() for p in self.model.parameters() if p.requires_grad)))
        self.pruned_heads = None

    def _init_model(self):
        print('[Pretrainer] - Initializing model...')
        self.model_config = Wav2Vec2Config(self.upstream_config['model'])
        self.model = Wav2Vec2Model(self.model_config, )

        # Do initialization from a checkpoint if needed
        if self.initial_weight:
            all_states = torch.load(self.initial_weight, map_location="cpu")
            print(f"loading model {all_states.keys()}")
            # If the attention heads have been pruned
            if 'Pruned_heads' in all_states:
                self.pruned_heads = all_states["Pruned_heads"]
                summarized = {}
                for layer_heads in self.pruned_heads:
                    for layer in layer_heads:
                        summarized[layer] = summarized.get(layer, 0) + len(layer_heads[layer])
                pruned_heads = summarized

                for idx, layer in enumerate(self.model.encoder.layers):
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
            elif all_states['Args'].mode == 'row-pruning':
                for idx, layer in enumerate(self.model.encoder.layers):
                    ffn_embed_dim, embed_dim = all_states["model"][f"encoder.layers.{idx}.fc1.weight"].shape
                    bias = True
                    layer.fc1 = nn.Linear(embed_dim, ffn_embed_dim, bias=bias)
                    layer.fc2 = nn.Linear(ffn_embed_dim, embed_dim, bias=bias)
                    layer.fc1.reset_parameters()
                    layer.fc2.reset_parameters()
                
            # Initialize the pruning mask if needed 
            # (Do not support executing other compression methods on a weight pruned checkpoint)
            if 'Pruning' in all_states:
                params_to_prune, _ = get_params_to_prune(self.model)
                prune.global_unstructured(
                    params_to_prune,
                    pruning_method=prune.Identity,
                )
            try:             
                self.model.load_state_dict(all_states["model"])
                print(f'[Pretrainer] Load initilization model weight from {self.initial_weight}')
            except:
               raise NotImplementedError('Could not load the initilization weight')

    def load_model(self, init_ckpt):
        assert 'model' in init_ckpt
        if self.multi_gpu:
            self.model.module.load_state_dict(init_ckpt['model'])
        else:
            self.model.load_state_dict(init_ckpt['model'])

    def add_state_to_save(self, all_states):
        all_states['model'] = self.model.state_dict() if not self.multi_gpu else self.model.module.state_dict()
        all_states['Upstream_Config'] = self.upstream_config
        if self.pruned_heads:
            all_states["Pruned_heads"] = self.pruned_heads
        return all_states
    
    def forward(self, data, global_step=0, log_step=1000):
        """
        Args:
            data:
                {"id": index, "source": wav, "label_list": labels}
            
            records:
                defaultdict(list), by appending contents into records,
                these contents can be averaged and logged on Tensorboard
                later by self.log_records every log_step
        Return:
            loss        
        """
        loss = 0.0 
        for key in data['net_input'].keys():
            data['net_input'][key] = data['net_input'][key].to(self.device)
        loss, sample_size= self.criterion.forward(self.model, data)
        
        return loss, sample_size