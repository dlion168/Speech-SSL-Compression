"""
    Model config and model structure of MelHuBERT. 
    Author: Tzu-Quan Lin (https://github.com/nervjack2)
    Reference: (https://github.com/s3prl/s3prl/tree/master/s3prl/upstream/distiller)
    Reference author: Heng-Jui Chang (https://github.com/vectominist)
"""
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Dictionary
from torch import nn
from module import TransformerEncoder, GradMultiply, ConvFeatureExtractionModel
from fairseq_code import compute_mask_indices, buffered_arange, index_put, is_xla_tensor

class MelHuBERTConfig:
    """
    Configuration class
    """

    def __init__(self, config: dict):
        # Input feature dimemsion 
        self.feat_emb_dim = int(config.get("feat_emb_dim", 40))
       
        # Positional embedding type
        self.pos_emb_type = str(config.get("pos_emb_type", "conv"))
        self.pos_conv_depth = int(config.get("pos_conv_depth", 1))
        self.conv_pos = int(config.get("conv_pos", 128))
        self.conv_pos_groups = int(config.get("conv_pos_groups", 16))

        # Transformer encoder
        self.encoder_layers = int(config.get("encoder_layers", 1))
        self.encoder_embed_dim = int(config.get("encoder_embed_dim", 768))
        self.encoder_ffn_embed_dim = int(config.get("encoder_ffn_embed_dim", 3072))
        self.encoder_attention_heads = int(config.get("encoder_attention_heads", 12))
        self.activation_fn = str(config.get("activation_fn", "gelu"))
        self.layer_norm_first = bool(config.get("layer_norm_first", False))
        self.attention_type = str(config.get("attention_type", "original"))
        # Output dimension 
        self.num_cluster = int(config.get("num_cluster", 512))
        self.final_dim = int(config.get("final_dim", 40))
        # Criterion (This two parameters would not be used in distillation mode)
        self.pred_masked_weight = float(config.get("pred_masked_weight", 1.0))
        self.pred_nomask_weight = float(config.get("pred_nomask_weight", 0.0))
        # Masking 
        self.mask_prob = float(config.get("mask_prob", 0.8))
        self.mask_length = int(config.get("mask_length", 10))
        self.mask_selection = str(config.get("mask_selection", 'static'))
        self.mask_other = float(config.get("mask_other", 0.0))
        self.no_mask_overlap = bool(config.get("no_mask_overlap", False))
        self.mask_min_space = int(config.get("mask_min_space", 1))

        self.skip_masked = bool(config.get("skip_masked", False))
        self.skip_nomask = bool(config.get("skip_nomask", True))

        self.learnable_mask_emb = bool(config.get("learnable_mask_emb", False))
        self.mask_before_proj = bool(config.get("mask_before_proj", True))
        # Dropout
        self.dropout = float(config.get("dropout", 0.1))
        self.attention_dropout = float(config.get("attention_dropout", 0.1))
        self.activation_dropout = float(config.get("activation_dropout", 0.1))
        self.encoder_layerdrop = float(config.get("encoder_layerdrop", 0.0))

class MelHuBERTModel(nn.Module):

    def __init__(self, model_config: MelHuBERTConfig):
        super().__init__()

        self.model_config = model_config

        self.n_encoder_layers = model_config.encoder_layers
        print(
            f"[MelHuBERTModel] - Encoder layer = {self.n_encoder_layers}"
        )

        self.pre_extract_proj = (
            nn.Linear(model_config.feat_emb_dim,  model_config.encoder_embed_dim)
            if model_config.feat_emb_dim != model_config.encoder_embed_dim
            else None
        )

        if model_config.encoder_layers > 0:
            self.encoder = TransformerEncoder(model_config)
        else:
            self.encoder = nn.GELU()
        
        if self.model_config.learnable_mask_emb:
            if not self.model_config.mask_before_proj: 
                self.mask_emb = nn.Parameter(
                    torch.FloatTensor(model_config.encoder_embed_dim).uniform_().to('cuda')
                )
            else:
                self.mask_emb = nn.Parameter(
                    torch.FloatTensor(model_config.feat_emb_dim).uniform_().to('cuda')
                )
        else:
            if not self.model_config.mask_before_proj:
                self.mask_emb = 0
            else:
                self.mask_emb = 0

        self.final_proj = nn.Linear(model_config.encoder_embed_dim, model_config.num_cluster)

    def apply_mask(self, x, padding_mask, teacher_mask_indices):
        """
        teacher_mask_indices: only for distillation mode 
        """
        B, T, C = x.shape
        if self.model_config.mask_prob > 0:
            if teacher_mask_indices != None:
                mask_indices = teacher_mask_indices
            else:
                mask_indices = compute_mask_indices(
                    (B, T),
                    padding_mask,
                    self.model_config.mask_prob,
                    self.model_config.mask_length,
                    self.model_config.mask_selection,
                    self.model_config.mask_other,
                    min_masks=2,
                    no_overlap=self.model_config.no_mask_overlap,
                    min_space=self.model_config.mask_min_space,
                    require_same_masks=False
                )
                mask_indices = torch.from_numpy(mask_indices).to(x.device)

            x[mask_indices] = self.mask_emb
        else:
            mask_indices =  None

        return x, mask_indices

    def forward(self, 
        feat, 
        pad_mask, 
        cluster_label=None, 
        no_pred=False,
        mask=False, 
        get_hidden=False, 
        teacher_mask_indices=None
    ):
        """
        Forward function
        Input:
            feat (FloatTensor): B x T_wave x D
            pad_mask (BoolTensor): B x T_wave
        """
        # Masking before projection 
        if mask and self.model_config.mask_before_proj:
            input_feat, mask_indices = self.apply_mask(feat, ~pad_mask.bool(), teacher_mask_indices)
        else:
            input_feat = feat
            mask_indices = torch.full(pad_mask.shape, False)

        pre_feat = input_feat
        if self.pre_extract_proj != None:
            pre_feat = self.pre_extract_proj(input_feat)
        
        # Masking after projection 
        if mask and not self.model_config.mask_before_proj:
            x, mask_indices = self.apply_mask(pre_feat, ~pad_mask.bool(), teacher_mask_indices)
        else:
            x = pre_feat
            mask_indices = mask_indices

        mask_indices = mask_indices.to(x.device)

        # implementation of causal attention
        if self.model_config.attention_type == "causal":
            seq_len = x.shape[1]
            attn_mask = torch.zeros(seq_len, seq_len)
            for idx in range(seq_len):
                attn_mask[idx, :idx+1] = 1
            attn_mask = torch.FloatTensor(attn_mask).to(
                device=x.device, dtype=torch.float32
            )  # (seq_len, seq_len)
            attn_mask = ~attn_mask.bool()
        else:
            attn_mask = None
        
        layer_hiddens = []
        if self.model_config.encoder_layers > 0:
            hidden, layer_hiddens = self.encoder(
                x, ~pad_mask.bool(), get_hidden=get_hidden, attn_mask=attn_mask
            )
        else:
            hidden = self.encoder(x)
        
        if no_pred:
            return hidden, None, None, None, None, layer_hiddens, pre_feat

        assert cluster_label != None

        if not self.model_config.skip_masked:
            masked_indices = torch.logical_and(pad_mask.bool(), mask_indices)
            logit_m = self.final_proj(hidden[masked_indices])  # (num_masked, dim) -> (num_masked, num_cluster)      
            label_m = cluster_label[masked_indices]
        else:
            logit_m = None
            label_m = None

        if not self.model_config.skip_nomask:
            nomask_indices = torch.logical_and(pad_mask.bool(), ~mask_indices)
            logit_u = self.final_proj(hidden[nomask_indices])  # (num_unmask, dim) -> (num_unmask, num_cluster)
            label_u = cluster_label[nomask_indices]  
        else:
            logit_u = None
            label_u = None
        
        return hidden, logit_m, logit_u, label_m, label_u, layer_hiddens, pre_feat, mask_indices

class HuBERTConfig:
    """
    Configuration class
    """

    def __init__(self, config: dict):
        self.extractor_mode = str(config.get("extractor_mode", "default"))
        
        # Transformer encoder
        self.encoder_layers = int(config.get("encoder_layers", 12))
        self.encoder_embed_dim = int(config.get("encoder_embed_dim", 768))
        self.encoder_ffn_embed_dim = int(config.get("encoder_ffn_embed_dim", 3072))
        self.encoder_attention_heads = int(config.get("encoder_attention_heads", 12))
        self.activation_fn = str(config.get("activation_fn", "gelu"))
        self.layer_type = str(config.get("layer_type", "transformer"))
        
        # Dropouts
        self.dropout = float(config.get("dropout", 0.1))
        self.attention_dropout = float(config.get("attention_dropout", 0.1))
        self.activation_dropout = float(config.get("activation_dropout", 0.0))
        self.encoder_layerdrop = float(config.get("encoder_layerdrop", 0.0))
        self.dropout_input = float(config.get("dropout_input", 0.0))
        self.dropout_features = float(config.get("dropout_features", 0.0))

        # Other parameters
        self.final_dim = int(config.get("final_dim", 0))
        self.untie_final_proj = bool(config.get("untie_final_proj", False))
        self.layer_norm_first = bool(config.get("layer_norm_first", False))
        self.conv_feature_layers = eval(config.get("conv_feature_layers", "[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2"))
        self.conv_bias = bool(config.get("conv_bias", False))
        self.logit_temp = float(config.get("logit_temp", 0.1))
        self.target_glu = bool(config.get("target_glu", False))
        self.feature_grad_mult = float(config.get("feature_grad_mult", 1.0))
        
        # Masking
        self.mask_length = int(config.get("mask_length", 10))
        self.mask_prob = float(config.get("mask_prob", 0.65))
        self.mask_selection = str(config.get("mask_selection", 'static'))
        self.mask_other = float(config.get("mask_other", 0))
        self.no_mask_overlap = bool(config.get("no_mask_overlap", False))
        self.mask_min_space = int(config.get("mask_min_space", 1))

        # Channel Masking
        self.mask_channel_length = int(config.get("mask_channel_length", 10))
        self.mask_channel_prob = float(config.get("mask_channel_prob", 0.0))
        self.mask_channel_selection = str(config.get("mask_channel_selection", "static"))
        self.mask_channel_other = float(config.get("mask_channel_other", 0))
        self.no_mask_channel_overlap = bool(config.get("no_mask_channel_overlap", False))
        self.mask_channel_min_space = int(config.get("mask_channel_min_space", 1))

        # Positional Embeddings
        self.conv_pos = int(config.get("conv_pos", 128))
        self.conv_pos_groups = int(config.get("conv_pos_groups", 16))
        self.conv_pos_batch_norm = bool(config.get("conv_pos_batch_norm", False))

        self.latent_temp = tuple(map(float, config.get("latent_temp", (2, 0.5, 0.999995))))

        # Loss Computation
        self.skip_masked = bool(config.get("skip_masked", False))
        self.skip_nomask = bool(config.get("skip_nomask", False))

        self.checkpoint_activations = bool(config.get("checkpoint_activations", False))

        # FP16 Optimization
        self.required_seq_len_multiple = int(config.get("required_seq_len_multiple", 2))

class HuBERTModel(nn.Module):
    def __init__(self,
                 cfg: HuBERTConfig,
                 dictionaries: List[Dictionary],):
        super().__init__()
        
        feature_enc_layers = eval(cfg.conv_feature_layers)  # noqa
        self.embed = feature_enc_layers[-1][0]
        self.feature_extractor = ConvFeatureExtractionModel(
            conv_layers=feature_enc_layers,
            dropout=0.0,
            mode=cfg.extractor_mode,
            conv_bias=cfg.conv_bias,
        )
        feature_ds_rate = np.prod([s for _, _, s in feature_enc_layers])
        self.feat2tar_ratio = cfg.label_rate * feature_ds_rate / 16000 # task_cfg.sample_rate

        self.post_extract_proj = (
            nn.Linear(self.embed, cfg.encoder_embed_dim)
            if self.embed != cfg.encoder_embed_dim
            else None
        )

        self.mask_prob = cfg.mask_prob
        self.mask_selection = cfg.mask_selection
        self.mask_other = cfg.mask_other
        self.mask_length = cfg.mask_length
        self.no_mask_overlap = cfg.no_mask_overlap
        self.mask_min_space = cfg.mask_min_space

        self.mask_channel_prob = cfg.mask_channel_prob
        self.mask_channel_selection = cfg.mask_channel_selection
        self.mask_channel_other = cfg.mask_channel_other
        self.mask_channel_length = cfg.mask_channel_length
        self.no_mask_channel_overlap = cfg.no_mask_channel_overlap
        self.mask_channel_min_space = cfg.mask_channel_min_space

        self.dropout_input = nn.Dropout(cfg.dropout_input)
        self.dropout_features = nn.Dropout(cfg.dropout_features)

        self.feature_grad_mult = cfg.feature_grad_mult
        self.logit_temp = cfg.logit_temp
        self.skip_masked = cfg.skip_masked
        self.skip_nomask = cfg.skip_nomask
        
        final_dim = cfg.final_dim if cfg.final_dim > 0 else cfg.encoder_embed_dim
        
        self.mask_emb = nn.Parameter(
            torch.FloatTensor(cfg.encoder_embed_dim).uniform_()
        )

        self.encoder = TransformerEncoder(cfg)
        self.layer_norm = nn.LayerNorm(self.embed)

        self.target_glu = None
        if cfg.target_glu:
            self.target_glu = nn.Sequential(
                nn.Linear(final_dim, final_dim * 2), nn.GLU()
            )

        self.untie_final_proj = cfg.untie_final_proj
        if self.untie_final_proj:
            self.final_proj = nn.Linear(
                cfg.encoder_embed_dim, final_dim * len(dictionaries)
            )
        else:
            self.final_proj = nn.Linear(cfg.encoder_embed_dim, final_dim)
        
        # modules below are not needed during fine-tuning
        if any([d is None for d in dictionaries]):
            print("cannot find dictionary. assume will be used for fine-tuning")
        else:
            self.num_classes = [len(d) for d in dictionaries]
            self.label_embs_concat = nn.Parameter(
                torch.FloatTensor(sum(self.num_classes), final_dim)
            )
            nn.init.uniform_(self.label_embs_concat)

    def apply_mask(self, x, padding_mask):
        B, T, C = x.shape
        if self.mask_prob > 0:
            mask_indices = compute_mask_indices(
                (B, T),
                padding_mask,
                self.mask_prob,
                self.mask_length,
                self.mask_selection,
                self.mask_other,
                min_masks=2,
                no_overlap=self.no_mask_overlap,
                min_space=self.mask_min_space,
            )
            mask_indices = torch.from_numpy(mask_indices).to(x.device)
            x[mask_indices] = self.mask_emb
        else:
            mask_indices = None
        return x, mask_indices
    
    def compute_nce(self, x, pos, negs):
        neg_is_pos = (pos == negs).all(-1)
        pos = pos.unsqueeze(0)
        targets = torch.cat([pos, negs], dim=0)

        logits = torch.cosine_similarity(x.float(), targets.float(), dim=-1).type_as(x)
        logits /= self.logit_temp
        if neg_is_pos.any():
            logits[1:][neg_is_pos] = float("-inf")
        logits = logits.transpose(0, 1)  # (num_x, num_cls+1)
        return logits
    
    def forward_features(self, source: torch.Tensor) -> torch.Tensor:
        if self.feature_grad_mult > 0:
            features = self.feature_extractor(source)
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.feature_grad_mult)
        else:
            with torch.no_grad():
                features = self.feature_extractor(source)
        return features
    
    @classmethod
    def build_model(cls, cfg: HuBERTConfig, task=None):
        """Build a new model instance."""

        return cls(cfg)
    
    def forward_targets(
        self,
        features: torch.Tensor,
        target_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Trim features to ensure labels exist and then get aligned labels
        feat_tsz = features.size(2)
        targ_tsz = min([t.size(1) for t in target_list])
        if self.feat2tar_ratio * feat_tsz > targ_tsz:
            feat_tsz = int(targ_tsz / self.feat2tar_ratio)
            features = features[..., :feat_tsz]
        target_inds = torch.arange(feat_tsz).float() * self.feat2tar_ratio
        target_list = [t[:, target_inds.long()] for t in target_list]
        return features, target_list
    
    def forward_padding_mask(
        self,
        features: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        extra = padding_mask.size(1) % features.size(1)
        if extra > 0:
            padding_mask = padding_mask[:, :-extra]
        padding_mask = padding_mask.view(padding_mask.size(0), features.size(1), -1)
        padding_mask = padding_mask.all(-1)
        return padding_mask

    def forward(
        self,
        source: torch.Tensor,
        target_list: Optional[List[torch.Tensor]] = None,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = True,
        features_only: bool = False,
        output_layer: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """output layer is 1-based"""
        features = self.forward_features(source)
        if target_list is not None:
            features, target_list = self.forward_targets(features, target_list)

        features_pen = features.float().pow(2).mean()

        features = features.transpose(1, 2)
        features = self.layer_norm(features)
        unmasked_features = features.clone()

        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(features, padding_mask)

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        features = self.dropout_input(features)
        unmasked_features = self.dropout_features(unmasked_features)

        if mask:
            x, mask_indices = self.apply_mask(features, padding_mask, target_list)
        else:
            x = features
            mask_indices = None

        # feature: (B, T, D), float
        # target: (B, T), long
        # x: (B, T, D), float
        # padding_mask: (B, T), bool
        # mask_indices: (B, T), bool
        x, _ = self.encoder(
            x,
            padding_mask=padding_mask,
            layer=None if output_layer is None else output_layer - 1,
        )

        if features_only:
            return {"x": x, "padding_mask": padding_mask, "features": features}

        def compute_pred(proj_x, target, label_embs):
            # compute logits for the i-th label set
            y = torch.index_select(label_embs, 0, target.long())
            negs = label_embs.unsqueeze(1).expand(-1, proj_x.size(0), -1)
            if self.target_glu:
                y = self.target_glu(y)
                negs = self.target_glu(negs)
            # proj_x: (S, D)
            # y: (S, D)
            # negs: (Neg, S, D)
            return self.compute_nce(proj_x, y, negs)

        label_embs_list = self.label_embs_concat.split(self.num_classes, 0)

        if not self.skip_masked:
            masked_indices = torch.logical_and(~padding_mask, mask_indices)
            proj_x_m = self.final_proj(x[masked_indices])
            if self.untie_final_proj:
                proj_x_m_list = proj_x_m.chunk(len(target_list), dim=-1)
            else:
                proj_x_m_list = [proj_x_m for _ in range(len(target_list))]
            logit_m_list = [
                compute_pred(proj_x_m, t[masked_indices], label_embs_list[i])
                for i, (proj_x_m, t) in enumerate(zip(proj_x_m_list, target_list))
            ]
        else:
            logit_m_list = [None for _ in target_list]

        if not self.skip_nomask:
            nomask_indices = torch.logical_and(~padding_mask, ~mask_indices)
            proj_x_u = self.final_proj(x[nomask_indices])
            if self.untie_final_proj:
                proj_x_u_list = proj_x_u.chunk(len(target_list), dim=-1)
            else:
                proj_x_u_list = [proj_x_u for _ in range(len(target_list))]

            logit_u_list = [
                compute_pred(proj_x_u, t[nomask_indices], label_embs_list[i])
                for i, (proj_x_u, t) in enumerate(zip(proj_x_u_list, target_list))
            ]
        else:
            logit_u_list = [None for _ in target_list]

        result = {
            "logit_m_list": logit_m_list,
            "logit_u_list": logit_u_list,
            "padding_mask": padding_mask,
            "features_pen": features_pen,
        }
        return result

    def extract_features(
        self,
        source: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = False,
        ret_conv: bool = False,
        output_layer: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        res = self.forward(
            source,
            padding_mask=padding_mask,
            mask=mask,
            features_only=True,
            output_layer=output_layer,
        )
        feature = res["features"] if ret_conv else res["x"]
        return feature, res["padding_mask"]

    def get_logits(self, net_output, is_masked=True):
        if is_masked:
            logits_list = net_output["logit_m_list"]
        else:
            logits_list = net_output["logit_u_list"]
        logits_list = [x.float() for x in logits_list if x is not None]
        return logits_list

    def get_targets(self, net_output, is_masked=True):
        logits_list = self.get_logits(net_output, is_masked)
        targets_list = [x.new_zeros(x.size(0), dtype=torch.long) for x in logits_list]
        return targets_list

    def get_extra_losses(self, net_output):
        extra_losses = []
        names = []

        if "features_pen" in net_output:
            extra_losses.append(net_output["features_pen"])
            names.append("features_pen")

        return extra_losses, names

    def remove_pretraining_modules(self):
        self.target_glu = None
        self.final_proj = None

    
    
class Wav2Vec2Config:
    def __init__(self, config):
        # Feature Extractor
        self.extractor_mode = str(config.get("extractor_mode", "default"))

        # Encoder
        self.encoder_layers = int(config.get("encoder_layers", 12))
        self.encoder_embed_dim = int(config.get("encoder_embed_dim", 768))
        self.encoder_ffn_embed_dim = int(config.get("encoder_ffn_embed_dim", 3072))
        self.encoder_attention_heads = int(config.get("encoder_attention_heads", 12))
        self.activation_fn = str(config.get("activation_fn", "gelu"))
        self.layer_type = str(config.get("layer_type", "transformer"))

        # Dropouts
        self.dropout = float(config.get("dropout", 0.1))
        self.attention_dropout = float(config.get("attention_dropout", 0.1))
        self.activation_dropout = float(config.get("activation_dropout", 0.0))
        self.encoder_layerdrop = float(config.get("encoder_layerdrop", 0.0))
        self.dropout_input = float(config.get("dropout_input", 0.0))
        self.dropout_features = float(config.get("dropout_features", 0.0))

        # Other Parameters
        self.final_dim = int(config.get("final_dim", 0))
        self.layer_norm_first = bool(config.get("layer_norm_first", False))
        self.conv_feature_layers = eval(config.get("conv_feature_layers", "[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] + [(512,2,2)]"))
        self.conv_bias = bool(config.get("conv_bias", False))
        self.logit_temp = float(config.get("logit_temp", 0.1))
        self.same_quantizer = bool(config.get("same_quantizer", False))
        self.target_glu = bool(config.get("target_glu", False))
        self.feature_grad_mult = float(config.get("feature_grad_mult", 1.0))
        self.quantizer_depth = int(config.get("quantizer_depth", 1))
        self.quantizer_factor = int(config.get("quantizer_factor", 3))
        self.latent_vars = int(config.get("latent_vars", 320))
        self.latent_groups = int(config.get("latent_groups", 2))
        self.latent_dim = int(config.get("latent_dim", 0))

        # Masking
        self.mask_length = int(config.get("mask_length", 10))
        self.mask_prob = float(config.get("mask_prob", 0.65))
        self.mask_selection = str(config.get("mask_selection", "static"))
        self.mask_other = float(config.get("mask_other", 0))
        self.no_mask_overlap = bool(config.get("no_mask_overlap", False))
        self.mask_min_space = int(config.get("mask_min_space", 1))
        self.require_same_masks = bool(config.get("require_same_masks", True))
        self.mask_dropout = float(config.get("mask_dropout", 0.0))

        # Channel Masking
        self.mask_channel_length = int(config.get("mask_channel_length", 10))
        self.mask_channel_prob = float(config.get("mask_channel_prob", 0.0))
        self.mask_channel_before = False
        self.mask_channel_selection = str(config.get("mask_channel_selection", "static"))
        self.mask_channel_other = float(config.get("mask_channel_other", 0))
        self.no_mask_channel_overlap = bool(config.get("no_mask_channel_overlap", False))
        self.mask_channel_min_space = int(config.get("mask_channel_min_space", 1))

        # Negative Selection
        self.num_negatives = int(config.get("num_negatives", 100))
        self.negatives_from_everywhere = bool(config.get("negatives_from_everywhere", False))
        self.cross_sample_negatives = int(config.get("cross_sample_negatives", 0))
        self.codebook_negatives = int(config.get("codebook_negatives", 0))

        # Positional Embeddings
        self.conv_pos = int(config.get("conv_pos", 128))
        self.conv_pos_groups = int(config.get("conv_pos_groups", 16))
        self.pos_conv_depth = int(config.get("pos_conv_depth", 1))

        # Latent Temperature
        self.latent_temp = tuple(map(float, config.get("latent_temp", (2, 0.5, 0.999995))))

        # Other Parameters
        self.max_positions = int(config.get("max_positions", 100000))
        self.checkpoint_activations = bool(config.get("checkpoint_activations", False))

        # FP16 Optimization
        self.required_seq_len_multiple = int(config.get("required_seq_len_multiple", 2))
        self.crop_seq_to_multiple = int(config.get("crop_seq_to_multiple", 1))

class Wav2Vec2Model(nn.Module):
    def __init__(self, cfg: Wav2Vec2Config):
        super().__init__()
        self.cfg = cfg

        feature_enc_layers = eval(cfg.conv_feature_layers)
        self.embed = feature_enc_layers[-1][0]

        self.feature_extractor = ConvFeatureExtractionModel(
            conv_layers=feature_enc_layers,
            dropout=0.0,
            mode=cfg.extractor_mode,
            conv_bias=cfg.conv_bias,
        )

        self.post_extract_proj = (
            nn.Linear(self.embed, cfg.encoder_embed_dim)
            if self.embed != cfg.encoder_embed_dim and not cfg.quantize_input
            else None
        )

        self.crop_seq_to_multiple = cfg.crop_seq_to_multiple

        self.mask_prob = cfg.mask_prob
        self.mask_selection = cfg.mask_selection
        self.mask_other = cfg.mask_other
        self.mask_length = cfg.mask_length
        self.no_mask_overlap = cfg.no_mask_overlap
        self.mask_min_space = cfg.mask_min_space

        self.mask_channel_prob = cfg.mask_channel_prob
        self.mask_channel_before = cfg.mask_channel_before
        self.mask_channel_selection = cfg.mask_channel_selection
        self.mask_channel_other = cfg.mask_channel_other
        self.mask_channel_length = cfg.mask_channel_length
        self.no_mask_channel_overlap = cfg.no_mask_channel_overlap
        self.mask_channel_min_space = cfg.mask_channel_min_space

        self.dropout_input = nn.Dropout(cfg.dropout_input)
        self.dropout_features = nn.Dropout(cfg.dropout_features)

        self.feature_grad_mult = cfg.feature_grad_mult

        self.quantizer = None
        self.input_quantizer = None

        self.n_negatives = cfg.num_negatives
        self.cross_sample_negatives = cfg.cross_sample_negatives
        self.codebook_negatives = cfg.codebook_negatives
        self.negatives_from_everywhere = cfg.negatives_from_everywhere

        self.logit_temp = cfg.logit_temp

        final_dim = cfg.final_dim if cfg.final_dim > 0 else cfg.encoder_embed_dim

        self.project_q = nn.Linear(self.embed, final_dim)

        self.mask_emb = nn.Parameter(
            torch.FloatTensor(cfg.encoder_embed_dim).uniform_()
        )
        encoder_cls = TransformerEncoder

        self.encoder = encoder_cls(cfg)
        self.layer_norm = nn.LayerNorm(self.embed)

        self.target_glu = None
        if cfg.target_glu:
            self.target_glu = nn.Sequential(
                nn.Linear(final_dim, final_dim * 2), nn.GLU()
            )

        self.final_proj = nn.Linear(cfg.encoder_embed_dim, final_dim)

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        return state_dict

    @classmethod
    def build_model(cls, cfg: Wav2Vec2Config, task=None):
        """Build a new model instance."""

        return cls(cfg)

    def apply_mask(
        self,
        x,
        padding_mask,
        mask_indices=None,
        mask_channel_indices=None,
    ):
        B, T, C = x.shape

        if self.mask_channel_prob > 0 and self.mask_channel_before:
            mask_channel_indices = compute_mask_indices(
                (B, C),
                None,
                self.mask_channel_prob,
                self.mask_channel_length,
                self.mask_channel_selection,
                self.mask_channel_other,
                no_overlap=self.no_mask_channel_overlap,
                min_space=self.mask_channel_min_space,
            )
            mask_channel_indices = (
                torch.from_numpy(mask_channel_indices)
                .to(x.device)
                .unsqueeze(1)
                .expand(-1, T, -1)
            )
            x[mask_channel_indices] = 0

        if self.mask_prob > 0:
            if mask_indices is None:
                mask_indices = compute_mask_indices(
                    (B, T),
                    padding_mask,
                    self.mask_prob,
                    self.mask_length,
                    self.mask_selection,
                    self.mask_other,
                    min_masks=2,
                    no_overlap=self.no_mask_overlap,
                    min_space=self.mask_min_space,
                    require_same_masks=self.cfg.require_same_masks,
                    mask_dropout=self.cfg.mask_dropout,
                )
                mask_indices = torch.from_numpy(mask_indices).to(x.device)
            x = index_put(x, mask_indices, self.mask_emb)
        else:
            mask_indices = None

        return x, mask_indices

    def sample_negatives(self, y, num, padding_count=None):

        if self.n_negatives == 0 and self.cross_sample_negatives == 0:
            return y.new(0)

        bsz, tsz, fsz = y.shape
        y = y.view(-1, fsz)  # BTC => (BxT)C

        # FIXME: what happens if padding_count is specified?
        cross_high = tsz * bsz
        high = tsz - (padding_count or 0)
        with torch.no_grad():
            assert high > 1, f"{bsz,tsz,fsz}"

            if self.n_negatives > 0:
                tszs = (
                    buffered_arange(num)
                    .unsqueeze(-1)
                    .expand(-1, self.n_negatives)
                    .flatten()
                )

                neg_idxs = torch.randint(
                    low=0, high=high - 1, size=(bsz, self.n_negatives * num)
                )
                neg_idxs[neg_idxs >= tszs] += 1

            if self.cross_sample_negatives > 0:
                tszs = (
                    buffered_arange(num)
                    .unsqueeze(-1)
                    .expand(-1, self.cross_sample_negatives)
                    .flatten()
                )

                cross_neg_idxs = torch.randint(
                    low=0,
                    high=cross_high - 1,
                    size=(bsz, self.cross_sample_negatives * num),
                )
                cross_neg_idxs[cross_neg_idxs >= tszs] += 1

        if self.n_negatives > 0:
            neg_idxs = neg_idxs + (torch.arange(bsz).unsqueeze(1) * high)
        else:
            neg_idxs = cross_neg_idxs

        if self.cross_sample_negatives > 0 and self.n_negatives > 0:
            neg_idxs = torch.cat([neg_idxs, cross_neg_idxs], dim=1)

        negs = y[neg_idxs.view(-1)]
        negs = negs.view(
            bsz, num, self.n_negatives + self.cross_sample_negatives, fsz
        ).permute(
            2, 0, 1, 3
        )  # to NxBxTxC
        return negs, neg_idxs

    def compute_preds(self, x, y, negatives):

        neg_is_pos = (y == negatives).all(-1)
        y = y.unsqueeze(0)
        targets = torch.cat([y, negatives], dim=0)

        logits = torch.cosine_similarity(x.float(), targets.float(), dim=-1)
        logits = logits / self.logit_temp
        logits = logits.type_as(x)

        if is_xla_tensor(logits) or neg_is_pos.any():
            if not hasattr(self, "_inftensor"):
                fillval = -float(2**30)
                self._inftensor = (
                    torch.tensor(fillval).to(x.device)
                    if is_xla_tensor(logits)
                    else float("-inf")
                )
            logits[1:] = index_put(logits[1:], neg_is_pos, self._inftensor)

        return logits

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """
        Computes the output length of the convolutional layers
        """

        def _conv_out_length(input_length, kernel_size, stride):
            return torch.floor((input_length - kernel_size) / stride + 1)

        conv_cfg_list = eval(self.cfg.conv_feature_layers)

        for i in range(len(conv_cfg_list)):
            input_lengths = _conv_out_length(
                input_lengths, conv_cfg_list[i][1], conv_cfg_list[i][2]
            )

        return input_lengths.to(torch.long)

    def forward(
        self,
        source,
        padding_mask=None,
        mask=True,
        features_only=False,
        layer=None,
        mask_indices=None,
        mask_channel_indices=None,
        padding_count=None,
        corpus_key=None,
    ):

        if self.feature_grad_mult > 0:
            features = self.feature_extractor(source)
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.feature_grad_mult)
        else:
            with torch.no_grad():
                features = self.feature_extractor(source)

        features_pen = features.float().pow(2).mean()

        features = features.transpose(1, 2)
        features = self.layer_norm(features)
        unmasked_features = features.clone()

        if padding_mask is not None and padding_mask.any():
            input_lengths = (1 - padding_mask.long()).sum(-1)
            # apply conv formula to get real output_lengths
            output_lengths = self._get_feat_extract_output_lengths(input_lengths)

            padding_mask = torch.zeros(
                features.shape[:2], dtype=features.dtype, device=features.device
            )

            # these two operations makes sure that all values
            # before the output lengths indices are attended to
            padding_mask[
                (
                    torch.arange(padding_mask.shape[0], device=padding_mask.device),
                    output_lengths - 1,
                )
            ] = 1
            padding_mask = (1 - padding_mask.flip([-1]).cumsum(-1).flip([-1])).bool()
        else:
            padding_mask = None

        time_steps_to_drop = features.size(1) % self.crop_seq_to_multiple
        if time_steps_to_drop != 0:
            features = features[:, :-time_steps_to_drop]
            unmasked_features = unmasked_features[:, :-time_steps_to_drop]
            if padding_mask is not None:
                padding_mask = padding_mask[:, :-time_steps_to_drop]

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        features = self.dropout_input(features)
        unmasked_features = self.dropout_features(unmasked_features)

        num_vars = None
        code_ppl = None
        prob_ppl = None
        curr_temp = None

        if self.input_quantizer:
            q = self.input_quantizer(features, produce_targets=False)
            features = q["x"]
            num_vars = q["num_vars"]
            code_ppl = q["code_perplexity"]
            prob_ppl = q["prob_perplexity"]
            curr_temp = q["temp"]
            features = self.project_inp(features)

        if mask:
            x, mask_indices = self.apply_mask(
                features,
                padding_mask,
                mask_indices=mask_indices,
                mask_channel_indices=mask_channel_indices,
            )
            if not is_xla_tensor(x) and mask_indices is not None:
                # tpu-comment: reducing the size in a dynamic way causes
                # too many recompilations on xla.
                y = unmasked_features[mask_indices].view(
                    unmasked_features.size(0), -1, unmasked_features.size(-1)
                )
            else:
                y = unmasked_features
        else:
            x = features
            y = unmasked_features
            mask_indices = None

        x, layer_results = self.encoder(
            x, padding_mask=padding_mask, layer=layer, corpus_key=corpus_key
        )

        if features_only:
            return {
                "x": x,
                "padding_mask": padding_mask,
                "features": unmasked_features,
                "layer_results": layer_results,
            }

        if self.quantizer:
            if self.negatives_from_everywhere:
                q = self.quantizer(unmasked_features, produce_targets=False)
                y = q["x"]
                num_vars = q["num_vars"]
                code_ppl = q["code_perplexity"]
                prob_ppl = q["prob_perplexity"]
                curr_temp = q["temp"]
                y = self.project_q(y)

                negs, _ = self.sample_negatives(
                    y,
                    mask_indices[0].sum(),
                    padding_count=padding_count,
                )
                y = y[mask_indices].view(y.size(0), -1, y.size(-1))

            else:
                q = self.quantizer(y, produce_targets=False)
                y = q["x"]
                num_vars = q["num_vars"]
                code_ppl = q["code_perplexity"]
                prob_ppl = q["prob_perplexity"]
                curr_temp = q["temp"]

                y = self.project_q(y)

                negs, _ = self.sample_negatives(
                    y,
                    y.size(1),
                    padding_count=padding_count,
                )

            if self.codebook_negatives > 0:
                cb_negs = self.quantizer.sample_from_codebook(
                    y.size(0) * y.size(1), self.codebook_negatives
                )
                cb_negs = cb_negs.view(
                    self.codebook_negatives, y.size(0), y.size(1), -1
                )  # order doesnt matter
                cb_negs = self.project_q(cb_negs)
                negs = torch.cat([negs, cb_negs], dim=0)
        else:
            y = self.project_q(y)

            if self.negatives_from_everywhere:
                negs, _ = self.sample_negatives(
                    unmasked_features,
                    y.size(1),
                    padding_count=padding_count,
                )
                negs = self.project_q(negs)
            else:
                negs, _ = self.sample_negatives(
                    y,
                    y.size(1),
                    padding_count=padding_count,
                )

        if not is_xla_tensor(x):
            # tpu-comment: reducing the size in a dynamic way causes
            # too many recompilations on xla.
            x = x[mask_indices].view(x.size(0), -1, x.size(-1))

        if self.target_glu:
            y = self.target_glu(y)
            negs = self.target_glu(negs)

        x = self.final_proj(x)
        x = self.compute_preds(x, y, negs)

        result = {
            "x": x,
            "padding_mask": padding_mask,
            "features_pen": features_pen,
        }

        if prob_ppl is not None:
            result["prob_perplexity"] = prob_ppl
            result["code_perplexity"] = code_ppl
            result["num_vars"] = num_vars
            result["temp"] = curr_temp

        return result

    def quantize(self, x):
        assert self.quantizer is not None
        x = self.feature_extractor(x)
        x = x.transpose(1, 2)
        x = self.layer_norm(x)
        return self.quantizer.forward_idx(x)

    def extract_features(
        self, source, padding_mask, mask=False, layer=None, corpus_key=None
    ):
        res = self.forward(
            source,
            padding_mask,
            mask=mask,
            features_only=True,
            layer=layer,
            corpus_key=corpus_key,
        )
        return res

    def get_logits(self, net_output):
        logits = net_output["x"]
        logits = logits.transpose(0, 2)
        logits = logits.reshape(-1, logits.size(-1))
        return logits

    def get_targets(self, sample, net_output, expand_steps=True):
        x = net_output["x"]
        return x.new_zeros(x.size(1) * x.size(2), dtype=torch.long)

    def get_extra_losses(self, net_output):
        pen = []

        if "prob_perplexity" in net_output:
            pen.append(
                (net_output["num_vars"] - net_output["prob_perplexity"])
                / net_output["num_vars"]
            )

        if "features_pen" in net_output:
            pen.append(net_output["features_pen"])

        return pen

    def remove_pretraining_modules(self, last_layer=None):
        self.quantizer = None
        self.project_q = None
        self.target_glu = None
        self.final_proj = None

        if last_layer is not None:
            self.encoder.layers = nn.ModuleList(
                l for i, l in enumerate(self.encoder.layers) if i <= last_layer
            )
