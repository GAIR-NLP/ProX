from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import os
import re
import yaml


@dataclass
class TinyLMConfig:
    model: str = "tiny_LLaMA_0_7b"
    name: str = "tinyllama_0_7b"

    # data dir
    train_data_dir: Path = None
    val_data_dir: Optional[Path] = None
    out_dir: Path = Path("out") / name  # out_dir is a Path object

    # hy-params
    num_of_nodes: int = 1
    num_of_devices: int = 8
    global_batch_size: int = 1024
    learning_rate: float = 4e-4
    micro_batch_size: int = 16
    max_step: int = 250000

    # token estimation:
    # max_seq _global_batch_size_ max_step
    # 2000 _1024_ 250000 _2 ~= 1T

    warmup_steps: int = 2000
    log_step_interval: int = 10
    eval_iters: int = 100
    save_step_interval: int = 5000
    eval_step_interval: int = 5000
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    decay_lr: bool = True
    min_lr: float = 4e-5
    tie_embedding: bool = False

    is_constant_lr: bool = False
    need_to_warm: bool = True
    is_continue_pt: bool = False
    only_validate: bool = False
    curr_iter: int = 0
    reinit_optim: bool = True

    initial_checkpoint_dir: str = ""

    # WandB Config
    project: str = "prox"
    entity: str = "prox"
    name: str = "fineweb-pro"

    train_data_config: list = field(default_factory=lambda: [("train_star", 1.0)])
    val_data_config: list = field(default_factory=lambda: None)

    @property
    def batch_size(self) -> int:
        return self.global_batch_size // (self.num_of_devices * self.num_of_nodes)

    @property
    def gradient_accumulation_steps(self) -> int:
        return self.batch_size // self.micro_batch_size

    @property
    def warmup_iters(self) -> int:
        return self.warmup_steps * self.gradient_accumulation_steps

    @property
    def max_iters(self) -> int:
        return self.max_step * self.gradient_accumulation_steps

    @property
    def lr_decay_iters(self) -> int:
        return self.max_iters

    @property
    def log_iter_interval(self) -> int:
        return self.log_step_interval * self.gradient_accumulation_steps

    @classmethod
    def from_yaml(cls, yaml_file):
        with open(yaml_file, "r") as f:
            config_dict = yaml.safe_load(f)
        
        for key, value in config_dict.items():
            # if value contains env variable, get the value from the environment
            if value is None or not isinstance(value, str):
                continue
            if "$" in value: # xxxx/$env_name/xxxxx or xxx/${env_name}/xxxx
                env_var_pattern = re.compile(r'\$\{([^}]+)\}|\$([A-Za-z_][A-Za-z0-9_]*)')
                envvar_name = env_var_pattern.search(value).group(1) or env_var_pattern.search(value).group(2)
                envvar_value = os.getenv(envvar_name)
                if envvar_value is None:
                    raise ValueError(f"Environment variable {envvar_name} is not set")
                config_dict[key] = value.replace("${"+envvar_name+"}", envvar_value).replace("$"+envvar_name, envvar_value)
            
        
        print("*** Config ***")
        print(config_dict)
        print("*** &&&&&& ***")

        config_dict["train_data_dir"] = Path(config_dict["train_data_dir"])
        config_dict["val_data_dir"] = Path(config_dict["val_data_dir"]) if config_dict["val_data_dir"] else None
        config_dict["out_dir"] = Path(config_dict["out_dir"])

        return cls(**config_dict)

    @classmethod
    def to_dict(cls, config):
        """Convert the dataclass attributes and properties to a dictionary."""
        data = asdict(config)
        data["batch_size"] = config.batch_size
        data["gradient_accumulation_steps"] = config.gradient_accumulation_steps
        data["warmup_iters"] = config.warmup_iters
        data["max_iters"] = config.max_iters
        data["lr_decay_iters"] = config.lr_decay_iters
        data["log_iter_interval"] = config.log_iter_interval
        return data
