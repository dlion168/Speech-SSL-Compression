"""
MelHuBERT distillation interface 
"""

import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
from model import MelHuBERTModel, MelHuBERTDistillerModel
from model_config import MelHuBERTConfig, MelHuBERTDistillerConfig


def freeze_model(model):
    """Freeze all parameters in a model."""
    for param in model.parameters():
        param.requires_grad = False

class MelHuBERTDistiller(nn.Module):
    def __init__(self, upstream_config, initial_weight=None, device='cuda', multi_gpu=False, **kwargs):
        super(MelHuBERTDistiller, self).__init__()

        self.initial_weight = initial_weight
        self.device = device
        self.multi_gpu = multi_gpu

        self.upstream_config = upstream_config
        # Initialize the model 
        self._init_model()
        # Define distillation loss 
        
        if upstream_config["student"]["loss_type"] == "l1":
            self.loss = nn.L1Loss(reduction="none")
        elif upstream_config["student"]["loss_type"] == "l2":
            self.loss = nn.MSELoss(reduction="none")
        else:
            raise NotImplementedError(upstream_config["student"]["loss_type"])

        self.cosine_loss = upstream_config["student"]["cosine_loss"]
        if self.cosine_loss > 0:
            print("[DistillerForPretrain] - Enabled cosine similarity loss.")

        # Make multiple GPU training possible
        if self.multi_gpu:
            self.model = torch.nn.DataParallel(self.model)
            print('[Distiller] - Multi-GPU training Enabled: ' + str(torch.cuda.device_count()))
        print('[Distiller] - Number of parameters: ' + str(sum(p.numel() for p in self.model.parameters() if p.requires_grad)))

    def _init_model(self):    
        print('[Distiller] - Initializing model...')
        # Define student model architecture
        self.student_config = MelHuBERTDistillerConfig(self.upstream_config['student'])
        self.model = MelHuBERTDistillerModel(self.student_config)
        
        # Load student model's weight if existed
        if self.initial_weight:
            stu_all_states = torch.load(self.initial_weight, map_location="cpu")
            try:             
                self.model.load_state_dict(stu_all_states["model"])
                print(f'[Distiller] Load student initilization model weight from {self.initial_weight}')
            except:
               raise NotImplementedError('Could not load the initilization weight')

        # Define teacher model architecture
        self.teacher_config = MelHuBERTConfig(self.upstream_config['teacher'])
        self.teacher_model = MelHuBERTModel(self.teacher_config)
        
        # Load teacher model's weight
        assert 'init_path' in self.upstream_config['teacher'], 'Please specify teacher\'s weight by self.upstream_config["teacher"]["init_ckpt"]'
        all_states = torch.load(self.upstream_config['teacher']['init_path'], map_location="cpu")
        
        try:             
            self.teacher_model.load_state_dict(all_states["model"])
            print(f'[Distiller] - Load teacher model\'s weight from {self.initial_weight}')
        except:
            raise NotImplementedError('Could not load the teacher model\'s weight')
        self.teacher_model.encoder.layerdrop = 0
        print("[DistillerForPretrain] - Disabled teacher's encoder layerdrop")
        assert self.student_config.n_tasks <= self.teacher_config.encoder_layers, (
            self.student_config.n_tasks,
            self.teacher_config.encoder_layers,
        )
        # Initializing from teacher
        if self.student_config.initial_from_teacher:
            print("[Distiller] - Initializing from teacher")
            self.model.encoder.pos_conv.load_state_dict(
                self.teacher_model.encoder.pos_conv.state_dict()
            )
            for l in range(self.student_config.encoder_layers):
                self.model.encoder.layers[l].load_state_dict(
                    self.teacher_model.encoder.layers[l].state_dict()
                )
        freeze_model(self.teacher_model)

    def load_model(self, init_ckpt):
        assert 'model' in init_ckpt
        if self.multi_gpu:
            self.model.module.load_state_dict(init_ckpt['model'])
        else:
            self.model.load_state_dict(init_ckpt['model'])

    def add_state_to_save(self, all_states):
        all_states['model'] = self.model.state_dict() if not self.multi_gpu else self.model.module.state_dict()
        all_states['Upstream_Config'] = self.upstream_config
        return all_states
    
    def compute_loss(self, feat, pred, target, return_other=False):
        """
        Computes loss.
        Inputs:
            feat: B x T x D
            pred: B x N x T x D
            target: B x N x T x D
        """

        # Reconstruction loss
        assert pred.shape == target.shape, (pred.shape, target.shape)
        rec_loss = self.loss(pred, target)  # B x N x T x D

        if return_other:
            with torch.no_grad():
                rec_layer_loss = rec_loss.mean((0, 2, 3))
        else:
            rec_layer_loss = None

        rec_loss = rec_loss.mean()

        # Cosine similarity loss
        if self.cosine_loss > 0:
            sim_loss = -F.logsigmoid(F.cosine_similarity(pred, target, dim=-1))
            # B x N x T
            if return_other:
                with torch.no_grad():
                    sim_layer_loss = sim_loss.mean((0, 2))
            else:
                sim_layer_loss = None
            sim_loss = sim_loss.mean()
        else:
            sim_loss = 0
            sim_layer_loss = None

        # Feature loss
        feat_pen = feat.float().pow(2).mean()

        total_loss = (
            rec_loss
            + feat_pen * self.student_config.feat_pen_loss
            + sim_loss * self.cosine_loss
        )

        return total_loss, rec_loss, rec_layer_loss, feat_pen, sim_loss, sim_layer_loss

    
    def acc(self, outputs, labels):
        acc = torch.sum(torch.argmax(outputs, dim=1) == labels).item()
        total = len(labels)
        return acc, total

    # Interface
    def forward(self, data, global_step=0, log_step=1000, return_other = False, **kwargs):
        """
        Args:
            data:
                [audio feature, cluster id, padding mask, audio length]
            
            records:
                defaultdict(list), by appending contents into records,
                these contents can be averaged and logged on Tensorboard
                later by self.log_records every log_step
        Return:
            loss        
        """
        audio_feat, label, pad_mask, audio_len = data[0], data[1], data[2], data[3]
        audio_feat = audio_feat.to(self.device)
        label = label.to(self.device)
        pad_mask = pad_mask.to(self.device)
        
        with torch.no_grad():
            with torch.cuda.amp.autocast(False):
                _, _, _, _, _, teacher_hiddens, _, _ = self.teacher_model(audio_feat, pad_mask, label, mask=True, get_hidden=True)
            if self.student_config.task_emb_type == "none":
                teacher_hiddens = teacher_hiddens[self.student_config.n_tasks]
                teacher_hiddens = teacher_hiddens.unsqueeze(1)
            else:
                if self.student_config.task_emb_type in ["expand-last", "hnet", "self-hidden"]:
                    teacher_hiddens = [
                        teacher_hiddens[i-1]
                        for i in self.model.pred_layer_id
                    ]
                else:
                    teacher_hiddens = teacher_hiddens["hidden_states"][1:]
                teacher_hiddens = torch.stack(teacher_hiddens, dim=1)  # B x N x T x D

        feat, _, pred, _, _ = self.model(audio_feat, pad_mask, label)
        
        (
            total_loss,
            rec_loss,
            rec_layer_loss,
            feat_pen,
            sim_loss,
            sim_layer_loss,
        ) = self.compute_loss(feat, pred, teacher_hiddens, return_other)
        
        return total_loss, 1
    