import os
import re
from dataclasses import dataclass
from typing import Optional, TypeVar

import yaml

DataclassT = TypeVar("DataclassT")


@dataclass
class GentaskConfig:
    "Data Path"
    data_path: Optional[str] = None

    "Save Configs"
    save_path: Optional[str] = "./data/data_gen"
    save_name: Optional[str] = "test"
    save_interval: Optional[int] = None

    @staticmethod
    def from_yaml(filepath):
        """
        Load the configuration from a YAML file
        Args:
            filepath (str): Path to the YAML file
        """
        with open(filepath, "r") as file:
            config_dict = yaml.safe_load(file)

        # Create a new instance of GentaskConfig and update its variables from the loaded data
        config = GentaskConfig
        for key, value in config_dict.items():
            # if value contains env variable, get the value from the environment
            if value is None or  not isinstance(value, str):
                pass
            elif "$" in value: # xxxx/$env_name/xxxxx or xxx/${env_name}/xxxx
                env_var_pattern = re.compile(r'\$\{([^}]+)\}|\$([A-Za-z_][A-Za-z0-9_]*)')
                envvar_name = env_var_pattern.search(value).group(1) or env_var_pattern.search(value).group(2)
                envvar_value = os.getenv(envvar_name)
                if envvar_value is None:
                    raise ValueError(f"Environment variable {envvar_name} is not set")
                config_dict[key] = value.replace("${"+envvar_name+"}", envvar_value).replace("$"+envvar_name, envvar_value)
                
            if hasattr(config, key):
                setattr(config, key, config_dict[key])

        return config