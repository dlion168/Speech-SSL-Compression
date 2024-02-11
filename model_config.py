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

class HuBERTConfig:
    """
    Configuration class
    """

    def __init__(self, config: dict):
        self.label_rate = int(config.get("label_rate", 50))
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
        self.pos_emb_type = str(config.get("pos_emb_type", "conv"))
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
        self.quantize_targets = bool(config.get("quantize_targets", False))
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
        self.pos_emb_type = str(config.get("pos_emb_type", "conv"))
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

class MelHuBERTDistillerConfig:
    """
    Configuration class
    """

    def __init__(self, config: dict):
        # Input feature dimemsion 
        self.feat_emb_dim = int(config.get("feat_emb_dim", 40))

        # Convolutional relative positional encoding
        self.pos_emb_type = str(config.get("pos_emb_type", "conv"))
        self.pos_conv_depth = int(config.get("pos_conv_depth", 1))
        self.conv_pos = int(config.get("conv_pos", 128))
        self.conv_pos_groups = int(config.get("conv_pos_groups", 16))
        
        self.learnable_mask_emb = bool(config.get("learnable_mask_emb", False))
        self.mask_before_proj = bool(config.get("mask_before_proj", True))

        # Transformer encoder
        self.encoder_layers = int(config.get("encoder_layers", 1))
        self.encoder_embed_dim = int(config.get("encoder_embed_dim", 768))
        self.encoder_ffn_embed_dim = int(config.get("encoder_ffn_embed_dim", 3072))
        self.encoder_attention_heads = int(config.get("encoder_attention_heads", 12))
        self.activation_fn = str(config.get("activation_fn", "gelu"))
        self.layer_norm_first = bool(config.get("layer_norm_first", False))
        self.attention_type = str(config.get("attention_type", "original"))

        # Dropout
        self.dropout = float(config.get("dropout", 0.1))
        self.attention_dropout = float(config.get("attention_dropout", 0.1))
        self.activation_dropout = float(config.get("activation_dropout", 0.1))
        self.encoder_layerdrop = float(config.get("encoder_layerdrop", 0.0))

        # Output
        self.final_dim = int(config.get("final_dim", 768))
        self.out_layer_type = str(config.get("out_layer_type", "expand-last"))
        self.out_layer_inter_dim = int(config.get("out_layer_inter_dim", -1))

        # Task & loss
        self.n_tasks = int(config.get("n_tasks", 12))
        self.task_emb_type = str(config.get("task_emb_type", "expand-last"))
        self.task_emb_size = int(config.get("task_emb_size", 0))
        self.layer_emb_size = int(config.get("layer_emb_size", 0))
        self.loss_type = str(config.get("loss_type", "l1"))
        self.feat_pen_loss = float(config.get("feat_pen_loss", 0.0))
        self.cosine_loss = float(config.get("cosine_loss", 0.0))

        # When task_emb_type == 'expand-last' only
        self.pred_layer_id = list(
            config.get("pred_layer_id", range(1, self.n_tasks + 1))
        )

        # Initialization
        self.initial_from_teacher = bool(
            config.get("initial_from_teacher", False)
        )