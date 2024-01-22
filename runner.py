"""
    Training interface of MelHuBERT.
    Author: Tzu-Quan Lin (https://github.com/nervjack2)
    Reference: (https://github.com/s3prl/s3prl/blob/master/s3prl/pretrain/runner.py)
    Reference author: Andy T. Liu (https://github.com/andi611)
"""
import os
import math
import glob
import yaml
from tqdm import tqdm
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from fairseq.data import Dictionary
from datasets.melhubert_dataset import MelFeatDataset
from datasets.hubert_dataset import HubertDataset
from datasets.wav2vec2_dataset import FileAudioDataset
from task_config import Wav2vec2TaskConfig, HubertTaskConfig
from pytorch_code import prune
import importlib
from typing import List

class LabelEncoder(object):
    def __init__(self, dictionary: Dictionary) -> None:
        self.dictionary = dictionary

    def __call__(self, label: str) -> List[str]:
        return self.dictionary.encode_line(
            label,
            append_eos=False,
            add_if_not_exist=False,
        )

class Runner():
    def __init__(self, args, runner_config):
        self.args = args
        self.runner_config = runner_config
        self.datasets = dict()
        self.dataset_to_epoch_iter = dict()
        self.logger = SummaryWriter(args.expdir)                                                     
        self.upstream_config = yaml.load(open(self.args.upstream_config, 'r'), Loader=yaml.FullLoader)
        self.upstream_pretrainer = self._get_upstream()
        self.batch_size = 1 if self.args.upstream == 'melhubert' else int(self.runner_config['pretrain_expert']['datarc']['train_batch_size'])

        # Assert the dimension of input projection layer
        if self.args.upstream == 'melhubert':
            if self.args.frame_period == 20:
                assert self.upstream_config['melhubert']['feat_emb_dim'] == 80, f'Feature embedding dimension should be {80} when the frame period is {20}'
            elif self.args.frame_period == 10:
                assert self.upstream_config['melhubert']['feat_emb_dim'] == 40, f'Feature embedding dimension should be {40} when the frame period is {10}'
            
        # Mode of pre-training
        if args.mode == 'melhubert':
            print(f'[Runner] Mode: Pre-training {self.args.upstream}')
            from upstream.melhubert.mh_utils import MelHuBERTTools
            self.mh_tools = MelHuBERTTools(
                self.args,
                self.runner_config,
                self.upstream_config,
                self.upstream_pretrainer
            )
            self.save_every_x_epochs = self.mh_tools.save_every_x_epochs
        elif args.mode == 'weight-pruning':
            print(f'[Runner] Mode: weight-pruning on {self.args.upstream}')
            from weight_pruning.wp_utils import WeightPruningTools
            self.wp_tools = WeightPruningTools(
                self.args,
                self.runner_config,
                self.upstream_config,
                self.upstream_pretrainer,
                self.args.initial_weight
            )

            self.total_prune_step = self.wp_tools.n_iters
            self.prune_steps = self.wp_tools.prune_steps
            self.period = self.wp_tools.period
            assert len(self.prune_steps) == self.total_prune_step, 'The length of pruning interval should equal to the total pruning steps' 
        elif args.mode == 'head-pruning':
            print(f'[Runner] Mode: {self.runner_config["prune"]["metric"]} head-pruning on {self.args.upstream}')
            from head_pruning.hp_utils import HeadPruningTools, set_prune_interval
            self.hp_tools = HeadPruningTools(
                self.args,
                self.runner_config,
                self.upstream_config,
                self.upstream_pretrainer
            )
            self.total_prune_step = self.runner_config['prune']['total_steps']
            self.prune_steps = set_prune_interval(
                prune_interval=self.runner_config['prune']['interval'],
                warm_up_steps=self.runner_config['prune']['warm_up'],  
                total_prune_steps=self.runner_config['prune']['total_steps']
            )
            assert len(self.prune_steps) == self.total_prune_step, 'The length of pruning interval should equal to the total pruning steps' 
        elif args.mode == 'row-pruning':
            print(f'[Runner] Mode: row-pruning on {self.args.upstream}')
            from row_pruning.rp_utils import RowPruningTools, set_prune_interval
            self.row_tools = RowPruningTools(
                self.args,
                self.runner_config,
                self.upstream_config,
                self.upstream_pretrainer
            )
            self.total_prune_step = self.runner_config['prune']['total_steps']
            self.prune_steps = set_prune_interval(
                prune_interval=self.runner_config['prune']['interval'],
                warm_up_steps=self.runner_config['prune']['warm_up'],  
                total_prune_steps=self.runner_config['prune']['total_steps']
            )
            assert len(self.prune_steps) == self.total_prune_step, 'The length of pruning interval should equal to the total pruning steps' 
        elif args.mode == 'distillation':
            print(f'[Runner] Mode: distillation on MelHuBERT')
            from upstream.melhubert_distiller.pretrain_expert import MelHuBERTDistiller
            from upstream.melhubert.mh_utils import MelHuBERTTools
            self.upstream_pretrainer = MelHuBERTDistiller(
                self.upstream_config,
                self.args.initial_weight,
                self.args.device,
                self.args.multi_gpu,).to(self.args.device)
            self.mh_tools = MelHuBERTTools(
                self.args,
                self.runner_config,
                self.upstream_config,
                self.upstream_pretrainer
            )
            self.save_every_x_epochs = self.mh_tools.save_every_x_epochs
        else:
            print('We do not support this mode currently.')

    def _get_upstream(self):
        module_path = f'upstream.{self.args.upstream}'
        Upstream = getattr(importlib.import_module(module_path), 'UpstreamPretrainExpert')
        if self.args.upstream == 'hubert':
            dicts = self.load_dictionaries()
            upstream = Upstream(self.upstream_config,
                                self.args.initial_weight,
                                self.args.device,
                                self.args.multi_gpu,
                                dicts = dicts
                                ).to(self.args.device)
        else :
            upstream = Upstream(self.upstream_config,
                    self.args.initial_weight,
                    self.args.device,
                    self.args.multi_gpu
                    ).to(self.args.device)

        assert hasattr(upstream, 'forward')
        assert hasattr(upstream, 'load_model')
        assert hasattr(upstream, 'add_state_to_save')
        return upstream

    def _get_optimizer(self, model):
        from torch.optim import Adam
        optimizer = Adam(model.parameters(), 
                         lr = float(self.runner_config.get('lr', 0.001)),
                         betas = tuple(self.runner_config.get('betas', (0.9, 0.999))),
                         eps = float(self.runner_config.get('eps', 1.0e-8)),
                         weight_decay = float(self.runner_config.get('weight_decay', 0)),
                    )    

        if self.args.init_optimizer_from_initial_weight:
            all_states = torch.load(self.args.initial_weight, map_location="cpu")
            init_optimizer = all_states["Optimizer"]
            try:
                optimizer.load_state_dict(init_optimizer)
                print(f'[Runner] Load initilization optimizer weight from {self.args.initial_weight}')
            except:
                raise NotImplementedError('Could not load the initilization weight of optimizer')

        return optimizer
    
    
    # def _get_lr_scheduler(self, optimizer, total_steps):

    #     from torch.optim.lr_scheduler import LambdaLR, PolynomialLR, SequentialLR
    #     scheduler1 = LambdaLR(optimizer, lr_lambda= lambda x: x/int(self.runner_config['lr_scheduler']['warmup_updates']))
    #     scheduler2 = PolynomialLR(optimizer, total_iters = total_steps - int(self.runner_config['lr_scheduler']['warmup_updates']))
    #     scheduler = SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[self.runner_config['lr_scheduler']['warmup_updates']])

    #     return scheduler
    
    def step_update(self, num_updates):
        """Update the learning rate after each update."""
        if self.cfg.warmup_updates > 0 and num_updates <= self.cfg.warmup_updates:
            self.warmup_factor = num_updates / float(self.cfg.warmup_updates)
            lr = self.warmup_factor * self.lr
        elif num_updates >= self.total_num_update:
            lr = self.end_learning_rate
        else:
            warmup = self.cfg.warmup_updates
            lr_range = self.lr - self.end_learning_rate
            pct_remaining = 1 - (num_updates - warmup) / (
                self.total_num_update - warmup
            )
            lr = lr_range * pct_remaining ** (self.power) + self.end_learning_rate
        return lr
    
    def load_dictionaries(self):
        label_dir = self.runner_config['task']['data'] if self.runner_config['task']['label_dir'] is None else self.runner_config['task']['label_dir']
        dictionaries = [
            Dictionary.load(f"{label_dir}/dict.{label}.txt")
            for label in self.runner_config['task']['labels']
        ]
        return [ dictionaries[0] ] if 'fine_tuning' in self.runner_config['task'] else dictionaries
    
    def get_label_dir(self) -> str:
        return self.runner_config['task']['data'] if self.runner_config['task']['label_dir'] is None else self.runner_config['task']['label_dir']

    def _get_dataloader(self,):
        if self.args.upstream == 'melhubert':
            dataset = MelFeatDataset(
                self.args.frame_period,
                self.upstream_config['task'],
                self.runner_config['datarc']['train_batch_size'],
                self.runner_config['datarc']['sets'],
                self.runner_config['datarc']['max_timestep'],
            )
        elif self.args.upstream == 'hubert':
            split = 'train'
            manifest = f"{self.runner_config['task']['data']}/{split}.tsv"
            dicts = self.load_dictionaries()
            pad_list = [dict.pad() for dict in dicts]
            eos_list = [dict.eos() for dict in dicts]
            procs = [LabelEncoder(dict) for dict in dicts]
            paths = [f"{self.get_label_dir()}/{split}.{l}" for l in self.runner_config['task']['labels']]
            task_cfg = HubertTaskConfig(self.runner_config['task'])
            dataset = HubertDataset(
                manifest,
                sample_rate=task_cfg.sample_rate,
                label_paths=paths,
                label_rates=task_cfg.label_rate,
                pad_list=pad_list,
                eos_list=eos_list,
                label_processors=procs,
                max_keep_sample_size=task_cfg.max_keep_size,
                min_keep_sample_size=task_cfg.min_sample_size,
                max_sample_size=task_cfg.max_sample_size,
                pad_audio=task_cfg.pad_audio,
                normalize=task_cfg.normalize,
                store_labels=False,
                random_crop=task_cfg.random_crop,
                single_target=task_cfg.single_target,
            )
        elif self.args.upstream == 'wav2vec2':
            split = 'train'
            task_cfg = Wav2vec2TaskConfig(self.args, self.runner_config['task'])
            data_path = task_cfg.data

            text_compression_level = task_cfg.text_compression_level

            compute_mask = getattr(task_cfg, "precompute_mask_config", None) is not None
            mask_args = {}
            manifest_path = os.path.join(data_path, "{}.tsv".format(split))
            dataset = FileAudioDataset(
                    manifest_path=manifest_path,
                    sample_rate=task_cfg.sample_rate,
                    max_sample_size=task_cfg.max_sample_size,
                    min_sample_size=task_cfg.min_sample_size,
                    pad=task_cfg.labels is not None or task_cfg.enable_padding,
                    normalize=task_cfg.normalize,
                    num_buckets=task_cfg.num_batch_buckets or int(task_cfg.tpu),
                    text_compression_level=text_compression_level,
                    compute_mask=compute_mask,
                    **mask_args,
                )
        dataloader = DataLoader(
            dataset, 
            batch_size = self.batch_size, # for bucketing
            shuffle=True, 
            num_workers=self.runner_config['pretrain_expert']['datarc']['num_workers'],
            drop_last=False, 
            pin_memory=True, 
            collate_fn=dataset.collate_fn
        )
        return dataloader

    def train(self):
        # Set model train mode
        self.upstream_pretrainer.train()
        # Prepare data
        gradient_accumulate_steps = self.runner_config['runner']['gradient_accumulate_steps']
        print('[Runner] - Accumulated batch size:', 
              self.runner_config['pretrain_expert']['datarc']['train_batch_size'] * gradient_accumulate_steps)
        # Get dataloader
        dataloader = self._get_dataloader()
        # Convert between pre-training epochs and total steps
        n_epochs = self.runner_config['runner']['n_epochs']
        if n_epochs > 0: 
            total_steps = int(n_epochs * len(dataloader.dataset) / gradient_accumulate_steps / self.batch_size)
            self.runner_config['runner']['total_steps'] = total_steps
            print(f'[Runner] - Training for {n_epochs} epochs, which is equivalent to {total_steps} steps')
        else:
            total_steps = self.runner_config['runner']['total_steps'] / self.batch_size
            n_epochs = int(total_steps * gradient_accumulate_steps * self.batch_size / len(dataloader.dataset))
            print(f'[Runner] - Training for {total_steps} steps, which is approximately {n_epochs} epochs')
    
        step_per_epoch = len(dataloader.dataset)//gradient_accumulate_steps

        # Check whether the pruning steps is smaller than the total amount of training steps
        if 'pruning' in self.args.mode:
            assert max(self.prune_steps) <= total_steps, f'Pruning steps {max(self.prune_steps)} should not be larger than the total training steps {total_steps}'
     
        assert self.runner_config['runner']['total_steps'] > self.runner_config['runner']['log_step']
        # Set optimizer
        optimizer = self._get_optimizer(self.upstream_pretrainer)
        # scheduler = self._get_lr_scheduler(optimizer=optimizer, total_steps=total_steps) if self.args.upstream != 'melhubert' else None
        # set progress bar
        pbar = tqdm(total=self.runner_config['runner']['total_steps'], dynamic_ncols=True, desc='overall')

        all_loss = 0
        batch_loss = 0
        global_step = 0
        backward_steps = 0
        prefix = f'{self.args.mode}/train-'

        while pbar.n < pbar.total:
            for data in tqdm(dataloader, dynamic_ncols=True, desc='train'):
                first_accu = (backward_steps % gradient_accumulate_steps == 0) 
                if self.args.mode in ['melhubert', 'distillation']:
                    # Save model for every x epochs in MelHuBERT pre-training mode
                    if (global_step % int(self.save_every_x_epochs * step_per_epoch) == 0) and first_accu:
                        num_epoch = global_step // step_per_epoch
                        self.mh_tools.save_model(optimizer, global_step, num_epoch)
                elif self.args.mode == 'weight-pruning':
                    if (global_step in self.prune_steps) and first_accu:
                        # Weight pruning
                        state = self.wp_tools.prune_api(optimizer, pbar.n, pbar.total)
                        if state == "not-converge":
                            pbar.total += self.period
                            self.prune_steps.append(max(self.prune_steps)+self.period)
                elif self.args.mode  == 'head-pruning':
                    if (global_step in self.prune_steps) and first_accu:
                        # Save model before pruning
                        self.hp_tools.save_model(optimizer, global_step)
                        # Head pruning
                        self.hp_tools.prune_api()       
                        # Redefine optimizer 
                        optimizer = self._get_optimizer(self.upstream_pretrainer)
                elif self.args.mode  == 'row-pruning':
                    if (global_step in self.prune_steps) and first_accu:
                        # Save model before pruning
                        self.row_tools.save_model(optimizer, global_step)
                        # Row pruning
                        self.row_tools.prune_api()       
                        # Redefine optimizer 
                        optimizer = self._get_optimizer(self.upstream_pretrainer)
                # try/except block for forward/backward
                try:
                    if pbar.n >= pbar.total:
                        break
                    global_step = pbar.n + 1

                    loss = self.upstream_pretrainer(
                        data,
                        global_step=global_step,
                        log_step=self.runner_config['runner']['log_step'],
                    )

                    if gradient_accumulate_steps > 1:
                        loss = loss / gradient_accumulate_steps
                    if self.args.multi_gpu:
                        loss = loss.sum()
                    loss.backward()

                except RuntimeError as e:
                    if 'CUDA out of memory' in str(e):
                        tqdm.write(f'[Runner] - CUDA out of memory at step {global_step}')
                        torch.cuda.empty_cache()
                        optimizer.zero_grad()
                        continue
                    else:
                        raise

                # Record loss
                loss_value = loss.item()
                all_loss += loss_value
                batch_loss += loss_value
                del loss
                
                # Whether to accumulate gradient
                backward_steps += 1
                if backward_steps % gradient_accumulate_steps > 0:
                    continue

                if self.args.mode == 'weight-pruning':
                    # Calculating smooth loss to exam converging during weight pruning
                    self.wp_tools.update_smooth_loss(batch_loss)
                    self.wp_tools.update_target_smooth_loss(global_step)
                    batch_loss = 0        
              
                # Gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(self.upstream_pretrainer.model.parameters(), self.runner_config['runner'].get('gradient_clipping', 0))
                if math.isnan(grad_norm):
                    tqdm.write(f'[Runner] - Error : grad norm is NaN at global step {global_step}')
                elif not math.isnan(grad_norm):
                    optimizer.step()
                    # if scheduler != None:
                    #     scheduler.step()
                optimizer.zero_grad()

                # Logging
                if global_step % self.runner_config['runner']['log_step'] == 0 or pbar.n == pbar.total -1:
                    # Log lossx
                    if global_step % self.runner_config['runner']['log_step'] == 0:
                        all_loss /= self.runner_config['runner']['log_step']
                    else:
                        all_loss /= (global_step % self.runner_config['runner']['log_step'])
                    # print(all_loss)
                    # if global_step == 10:
                    #     exit(0)
                    self.logger.add_scalar(f'{prefix}loss', all_loss, global_step=global_step)

                    all_loss = 0
                    # Log norm
                    self.logger.add_scalar(f'{prefix}gradient norm', grad_norm, global_step=global_step)
                # Save model at the last step
                if pbar.n == pbar.total-1:
                    if self.args.mode in ['melhubert', 'distillation']:
                        name = 'last-step.ckpt'
                        self.mh_tools.save_model(optimizer, global_step, num_epoch, name=name)
                    elif self.args.mode == 'weight-pruning':
                        name = 'last-step.ckpt'
                        self.wp_tools._save(optimizer, pbar.n, pbar.total, filename=name)
                    elif self.args.mode == 'head-pruning':
                        self.hp_tools.save_model(optimizer, global_step)
                    elif self.args.mode == 'row-pruning':
                        self.row_tools.save_model(optimizer, global_step)
                pbar.update(1)

        pbar.close()
