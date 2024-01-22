from typing import List, Optional

class HubertTaskConfig:
    """
    Configuration class
    """

    def __init__(self, config: dict):
        self.data = str(config.get("data", None))
        self.fine_tuning = bool(config.get("fine_tuning", False))
        self.labels = list[str](config.get("labels", ["ltr"]))
        self.label_dir = str(config.get("label_dir", None))
        self.label_rate = float(config.get("label_rate", -1.0))
        self.sample_rate = int(config.get("sample_rate", 16_000))
        self.normalize = bool(config.get("normalize", False))
        self.enable_padding = bool(config.get("enable_padding", False))
        self.max_keep_size = int(config["max_keep_size"]) if "max_keep_size" in config else None
        self.max_sample_size = int(config["max_sample_size"]) if "max_sample_size" in config else None
        self.min_sample_size = int(config["min_sample_size"]) if "min_sample_size" in config else None
        self.single_target = bool(config.get("single_target", False))
        self.random_crop = bool(config.get("random_crop", True))
        self.pad_audio = bool(config.get("pad_audio", False))