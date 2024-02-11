"""
MelHuBERT distillation interface 
"""

import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq_code import is_xla_tensor
from model import Wav2Vec2Model, Wav2Vec2Config

class Wav2vec2Distiller(nn.Module):
    def __init__(self, upstream_config, initial_weight=None, device='cuda', multi_gpu=False, **kwargs):
        super(Wav2vec2Distiller, self).__init__()

        self.initial_weight = initial_weight
        self.device = device
        self.multi_gpu = multi_gpu

        self.upstream_config = upstream_config
        # Initialize the model 
        self._init_model()
        # Define distillation loss 
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='sum')        
        self.loss_temp = self.upstream_config['loss_param']['T']
        self.loss_alpha = self.upstream_config['loss_param']['alpha']

        # Make multiple GPU training possible
        if self.multi_gpu:
            self.model = torch.nn.DataParallel(self.model)
            print('[Distiller] - Multi-GPU training Enabled: ' + str(torch.cuda.device_count()))
        print('[Distiller] - Number of parameters: ' + str(sum(p.numel() for p in self.model.parameters() if p.requires_grad)))

    def _init_model(self):    
        print('[Distiller] - Initializing model...')
        # Define student model architecture
        self.student_config = Wav2Vec2Config(self.upstream_config['student'])
        self.model = Wav2Vec2Model(self.student_config)
        # Define teacher model architecture
        self.teacher_config = Wav2Vec2Config(self.upstream_config['teacher'])
        self.teacher_model = Wav2Vec2Model(self.teacher_config)
        # Load teacher model's weight
        assert self.initial_weight, 'Please specify teacher\'s weight by -i argument'
        all_states = torch.load(self.initial_weight, map_location="cpu")
        try:             
            self.teacher_model.load_state_dict(all_states["model"])
            print(f'[Distiller] - Load teacher model\'s weight from {self.initial_weight}')
        except:
            raise NotImplementedError('Could not load the teacher model\'s weight')

        # Initializing from teacher
        if self.upstream_config['student']['initial_from_teacher']:
            print("[Distiller] - Initializing from teacher")
            self.model.encoder.pos_conv.load_state_dict(
                self.teacher_model.encoder.pos_conv.state_dict()
            )
            for l in range(self.student_config.encoder_layers):
                self.model.encoder.layers[l].load_state_dict(
                    self.teacher_model.encoder.layers[l].state_dict()
                )
            self.model.feature_extractor.load_state_dict(
                self.teacher_model.feature_extractor.state_dict()
            )
            if self.model.post_extract_proj is not None:
                self.model.post_extract_proj.load_state_dict(
                    self.teacher_model.post_extract_proj.state_dict()
                )

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
    
    def loss_fn_kd(self, stu_logits, labels, teacher_outputs, stu_net_output, sample_size, T=1, alpha=0.5, loss_weights=[0.1, 10]):
        # # Teacher's cross entropy loss
        # teacher_loss = self.loss(teacher_outputs, labels)
        # # Student's cross entropy loss
        # hard_loss = self.loss(stu_logits, labels)
        # # Student-Teacher KL divergence loss
        soft_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(stu_logits/T, dim=0),
                                F.softmax(teacher_outputs/T, dim=0))
        x = F.softmax(teacher_outputs/T, dim=0)
        tot = [0, 0, 0, 0, 0, 0, 0]
        for i, xi in enumerate(x):
            for j in range(7):
                tot[j] += xi[j]
        print(tot)
        # # Extra loss for codebook perplexity in wav2vec2
        # if loss_weights is not None:
        #     stu_extra_loss = []
        #     if "prob_perplexity" in stu_net_output:
        #         stu_extra_loss.append(
        #             (stu_net_output["num_vars"] - stu_net_output["prob_perplexity"])
        #             / stu_net_output["num_vars"]
        #         )
        #     if "features_pen" in stu_net_output:
        #         stu_extra_loss.append(stu_net_output["features_pen"])

        #     if len(loss_weights) == 1 and len(stu_extra_loss) != 1:
        #         loss_weights = [loss_weights[0]] * len(stu_extra_loss)
        #     assert len(stu_extra_loss) == len(
        #         loss_weights
        #     ), f"{len(stu_extra_loss)}, {len(loss_weights)}"
        #     for p, coef in zip(stu_extra_loss, loss_weights):
        #         if coef != 0 and p is not None:
        #             p = coef * p.float() * sample_size
        #             hard_loss += p
        
        # total_loss = (hard_loss * (1. - alpha)) + (soft_loss * alpha)
        # return total_loss, hard_loss, soft_loss, teacher_loss
        return soft_loss, 0, soft_loss, 0
    
    def acc(self, outputs, labels):
        acc = torch.sum(torch.argmax(outputs, dim=1) == labels).item()
        total = len(labels)
        return acc, total

    # Interface
    def forward(self, sample, reduce=True, **kwargs):
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
        for s in sample['net_input'].keys():
            sample['net_input'][s] = sample['net_input'][s].to(self.device)
        net_output = self.model(**sample["net_input"])
        logits = self.model.get_logits(net_output).float()
        
        t_net_output = self.teacher_model(**sample["net_input"], mask_indices = net_output['mask_indices'])
        t_logits = self.teacher_model.get_logits(t_net_output).float()
        target = self.teacher_model.get_targets(sample, t_net_output)
        
        if "sample_size" in sample:
            sample_size = sample["sample_size"]
        elif "mask_indices" in sample["net_input"]:
            sample_size = sample["net_input"]["mask_indices"].sum()
        else:
            sample_size = target.numel()

        loss = 0.0 
        hard_loss, soft_loss = 0.0, 0.0
        teacher_loss = 0.0
        
        all_loss, h_loss, s_loss, t_loss = self.loss_fn_kd(logits, target, t_logits, net_output, sample_size, T=self.loss_temp, alpha=self.loss_alpha)
        loss += all_loss
        hard_loss += h_loss
        soft_loss += s_loss
        teacher_loss += t_loss
        
        return loss, sample_size
    