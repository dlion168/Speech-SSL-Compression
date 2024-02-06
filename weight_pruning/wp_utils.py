import re 
import os 
import typing
import torch 
import numpy as np 
import random
import torch.nn as nn
from torch.optim import Optimizer
from pytorch_code import prune
from tqdm import tqdm

# Define group of parameters to be pruned 
def get_params_to_prune(upstream, bias=True):
    # Define the module to prune by regax
    prune_regax = r".*encoder\.layers\.[0-9]+\.((self_attn\.([qkv]|out)_proj)|fc[12])\.weight"

    params_to_prune = tuple()

    model = upstream.module if isinstance(upstream, nn.DataParallel) else upstream

    for layer in model.encoder.layers:
        params_to_prune = (
            *params_to_prune,
            # self attention layer
            (layer.self_attn.q_proj, "weight"),
            (layer.self_attn.k_proj, "weight"),
            (layer.self_attn.v_proj, "weight"),
            (layer.self_attn.out_proj, "weight"),
            # fc layer
            (layer.fc1, "weight"),
            (layer.fc2, "weight"),
        )
        if bias:
            params_to_prune = (
                *params_to_prune,
                # bias
                (layer.self_attn.q_proj, "bias"),
                (layer.self_attn.k_proj, "bias"),
                (layer.self_attn.v_proj, "bias"),
                (layer.self_attn.out_proj, "bias"),
                (layer.fc1, "bias"),
                (layer.fc2, "bias"),
            )
            
    def name_filter(name):
        return re.fullmatch(prune_regax, name)
            
    return params_to_prune, name_filter

def _resume_random_state(state):
    if state:
        random.setstate(state['random'])
        np.random.set_state(state['numpy'])
        torch.set_rng_state(state['torch'])
        torch.cuda.set_rng_state(state['torch.cuda'])

class WeightPruningTools():
    def __init__(self, args, runner_config, upstream_config, upstream, initial_weight):
        self.args = args
        self.runner_config = runner_config
        self.upstream_config = upstream_config
        self.upstream = upstream
        self.initial_weight = initial_weight

        self.prune_condition = self.runner_config["prune"]["pruning_condition"]
        self.prune_strategy = self.runner_config["prune"]["strategy"]
        # Set pruning-related parameters
        self.n_iters = self.runner_config["prune"].get("n_iters", 38)
        self.warmup = self.runner_config["prune"].get("warmup", 25000)
        if type(self.runner_config["prune"]["period"]) == int:
            self.period = self.runner_config["prune"].get("period", 25000)
        elif type(self.runner_config["prune"]["period"]) == list:
            self.period = self.runner_config["prune"]["period"]
        else:
            raise NotImplementedError
        self.period = self.runner_config["prune"].get("period", 25000)
        # assert self.warmup > 0 and self.period > 0, f"Do not set warmup and period to 0."
        assert self.warmup > 0, f"Do not set warmup and period to 0."
        self.avg_len = self.runner_config["prune"].get("average_length", 15000)
        self.con_tol = self.runner_config["prune"].get("converge_loss_tolerance", 0.001)
        # Setup pruning sparsity
        if type(self.runner_config["prune"]["sparsity"]) == float:
            self.sparsity = [self.runner_config["prune"]["sparsity"] * (n+1) / self.n_iters for n in range(self.n_iters)]
        elif type(self.runner_config["prune"]["sparsity"]) == list:
            self.sparsity = self.runner_config["prune"]["sparsity"]
        else:
            raise NotImplementedError
        # Define the pruning steps (only support fixed interval)
        if type(self.period) == int:
            self.prune_steps = list(self.warmup + (np.arange(self.n_iters) * self.period))
        else:
            self.prune_steps = list(self.warmup + np.array(self.period))
        assert len(self.prune_steps) == len(self.sparsity)
        # Smooth loss is used to exam converging during iteratively weight pruning 
        self.smooth_loss = None 
        self.tgt_smooth_loss = -float("inf")
        self.smooth_factor = self.runner_config['prune'].get('smooth_factor', 0.999)
        self.buffer_loss = [] 
        self.pruning_times = 0
        params_to_prune, _ = get_params_to_prune(self.upstream.model)
        prune.global_unstructured(
            params_to_prune,
            pruning_method=prune.Identity,
        )

        # Resume training 
        if self.initial_weight:
            all_states = torch.load(self.initial_weight, map_location="cpu")
            if "Pruning" in all_states:
                self.smooth_loss = all_states["Pruning"]["smooth_loss"]
                self.tgt_smooth_loss = all_states["Pruning"]["tgt_smooth_loss"]
                self.pruning_times = all_states["Pruning"]["pruning_times"]
            if "RandomState" in all_states:
                random_state = all_states["RandomState"]
                _resume_random_state(random_state)

        print("="*40 + "\n[Weight Pruning] - Pruning-related hyperparameters:")
        print(f"Pruning iterations: {self.n_iters}")
        print(f"Warmup steps: {self.warmup}")
        print(f"Pruning steps: {self.prune_steps}")
        print("="*40)

    def update_smooth_loss(self, batch_loss):
        if self.smooth_loss is not None:
            self.smooth_loss = self.smooth_loss * self.smooth_factor + batch_loss * (1-self.smooth_factor)
        elif len(self.buffer_loss) == 3:
            # First 100 steps after pruning 
            self.smooth_loss = sum(self.buffer_loss) / 3
            self.buffer_loss = []
        else:
            self.buffer_loss.append(batch_loss)

    def update_target_smooth_loss(self, global_step):
        # Recording smooth loss n steps before pruning to exam converging 
        if (self.prune_condition == "converge" and global_step > self.warmup and
                (global_step - self.warmup + self.avg_len) in self.prune_steps):
            self.tgt_smooth_loss = self.smooth_loss
        
    def prune_api(self, optimizer, global_step, total_step):
        if self.prune_condition == "converge" and self.tgt_smooth_loss - self.con_tol > self.smooth_loss:
            tqdm.write('[Weight Pruning] - Not converge, keep training')
            return "not-converge"
        # Save checkpoint before pruning
        fname_prefix = "mask-" if prune.is_pruned(self.upstream.model) else ""
        # filename = f'{fname_prefix}before-pruning-states-{global_step}.ckpt'
        cur_sparsity = 0 if self.pruning_times == 0 else self.sparsity[self.pruning_times-1]  
        filename = f'{fname_prefix}before-pruning-states-{global_step}-sparsity-{cur_sparsity}.ckpt'
        self._save(optimizer, global_step, total_step, filename)
        # Pruning
        params_to_prune, name_filter = get_params_to_prune(self.upstream.model)
        amount = self.sparsity[self.pruning_times]
        for module, name in params_to_prune:
            prune.remove(module, name)
        prune.global_unstructured(
            params_to_prune,
            pruning_method=getattr(prune, self.prune_strategy),
            amount=amount
        )
        tqdm.write(f"[Weight Pruning] - {self.pruning_times+1} iters of pruning at {global_step} steps")
        self.pruning_times += 1 
        self.smooth_loss = None 
        return "pruned"

    def _save(
        self,
        optimizer: Optimizer,
        global_step: int,
        total_step: int,
        filename: str,
        ):
        
        all_states = {
            'Optimizer': optimizer.state_dict(),
            'Step': global_step,
            'TotalStep': total_step,
            'Args': self.args,
            'Runner': self.runner_config,
            'Pruning': {
                'smooth_loss': self.smooth_loss,
                'tgt_smooth_loss': self.tgt_smooth_loss,
                'pruning_times': self.pruning_times,
            },
            'RandomState': {
                'random': random.getstate(),
                'numpy': np.random.get_state(),
                'torch': torch.get_rng_state(),
                'torch.cuda': torch.cuda.get_rng_state(),
            }
        }
        all_states = self.upstream.add_state_to_save(all_states)
        
        save_path = os.path.join(self.args.expdir, filename)
        tqdm.write(f'[Weight Pruning] - Save the checkpoint to: {save_path}')
        torch.save(all_states, save_path)