import yaml
import pathlib


def load_config(config_path=None):
    if config_path is None:
        root_path = pathlib.Path(__file__).parent.absolute()
        config_path = "{}/config.yaml".format(root_path)
    with open(config_path, 'r') as yaml_file:
        config_dict = yaml.safe_load(yaml_file)
        return Config(config_dict)


class Config:

    def __init__(self, config_dict):
        self._config_dict = config_dict

    def __getattribute__(self, key):
        if key == '_config_dict' or key not in self._config_dict:
            return super().__getattribute__(key)
        value = self._config_dict[key]
        if isinstance(value, dict):
            return Config(value)
        elif isinstance(value, list):
            return [Config(item) if isinstance(item, dict) else item for item in value]
        else:
            return value

    def __getitem__(self, key):
        return self.__getattribute__(key)

    def __contains__(self, item):
        return item in self._config_dict

    def to_dict(self):
        return self._config_dict


global_config = load_config()
