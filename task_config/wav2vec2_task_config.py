from typing import Dict

class Wav2vec2TaskConfig:
    """
    Configuration class
    """

    def __init__(self, args: dict, config: dict):
        # Data Configuration
        self.data = str(config.get("data", None))
        self.labels = str(config.get("labels", None))
        self.binarized_dataset = bool(config.get("binarized_dataset", False))
        self.sample_rate = int(config.get("sample_rate", 16000))
        self.normalize = bool(config.get("normalize", False))
        self.enable_padding = bool(config.get("enable_padding", False))
        self.max_sample_size = int(config.get("max_sample_size", None))
        self.min_sample_size = int(config.get("min_sample_size", None))
        self.num_batch_buckets = int(config.get("num_batch_buckets", 0))
        self.tpu = False
        self.text_compression_level = int(config.get("text_compression_level", 0))

        # Additional Configurations
        self.rebuild_batches = bool(config.get("rebuild_batches", True))
        self.precompute_mask_config = dict(config.get("precompute_mask_config", None)) if config.get("precompute_mask_config", None) else None
        self.post_save_script = str(config.get("post_save_script", None))
        self.subsample = float(config.get("subsample", 1))
        self.seed = int(args.seed)
