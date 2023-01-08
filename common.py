from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List

import yaml


def list_sorted_names_in_dir(dir_path: Path) -> List[str]:
    return sorted([m.name for m in dir_path.glob('*') if m.is_dir()])


class ObjectWithConf(ABC):
    """
    A class for objects with trackable configuration
    """

    @abstractmethod
    def get_conf(self) -> Dict[str, object]:
        """
        Gets the instance configuration with the parameters the instance was initialised with
        :return: Dictionary
        """
        pass

    def _save_conf(self, path: Path):
        with open(path, mode='w') as conf_file:
            yaml.dump(self.get_conf(), conf_file)

    @staticmethod
    def _load_conf(path: Path):
        with open(path, mode='r') as conf_file:
            return yaml.safe_load(conf_file)
