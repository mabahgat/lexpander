from pathlib import Path
from typing import List, Tuple

import pandas as pd
import yaml

from common import ObjectWithConf
from config import global_config


def get_dataset_root_path() -> Path:
    return Path(global_config.storage.root) / global_config.storage.datasets_subdir


def list_dataset_names(datasets_root_path: Path = get_dataset_root_path()) -> List[str]:
    """
    Returns a list of dictionaries
    :param datasets_root_path: Optional override for datasets root directory
    :return: Names as string list
    """
    return sorted([d.name for d in datasets_root_path.glob('*') if d.is_dir()])


class DatasetNotFoundError(FileNotFoundError):
    pass


class DictionaryDataset(ObjectWithConf):
    
    def __init__(self,
                 name: str,
                 train_df: pd.DataFrame = None,
                 valid_df: pd.DataFrame = None,
                 test_df: pd.DataFrame = None,
                 source_path: str = None,
                 overwrite_if_exists: bool = False,
                 datasets_root_path: Path = None):
        """
        New Instance of dictionary datasets that encapsulates dictionary data
        :param name: Required - dictionary unique identifier
        :param train_df:
        :param valid_df:
        :param test_df:
        :param source_path: Raw file path from which the passed dataframes were generated
        :param overwrite_if_exists:
        :param datasets_root_path: Overrides default root path
        """
        if name is None:
            raise ValueError('Dataset name can not be none')
        self._name = name
        self._datasets_root_path = datasets_root_path if datasets_root_path is not None else get_dataset_root_path()
        self._root_path, self._train_path, self._valid_path, self._test_path = self.__generate_paths()
        if train_df is not None and test_df is not None:
            self._train_df = train_df
            self._valid_df = valid_df
            self._test_df = test_df
            self.__source_path = source_path
            self._save(overwrite_if_exists)
        else:
            self._train_df, self._valid_df, self._test_df = self._load()
            self.__source_path = self.__load_conf()['source_file']

    def __generate_paths(self) -> Tuple[Path, Path, Path, Path]:
        root_path = self._datasets_root_path / self._name

        def get_file_path(set_type: str) -> Path:
            return root_path / '{}__{}.csv'.format(self._name, set_type)
        train_path = get_file_path('train')
        valid_path = get_file_path('valid')
        test_path = get_file_path('test')

        return root_path, train_path, valid_path, test_path

    def __get_conf_path(self):
        return self._root_path / '{}__conf.yaml'.format(self._name)

    def _save(self, overwrite: bool):
        if not self._root_path.exists():
            self._root_path.mkdir(parents=True, exist_ok=overwrite)
        self._train_df.to_csv(self._train_path)
        if self._valid_path is not None:
            self._valid_df.to_csv(self._valid_path)
        self._test_df.to_csv(self._test_path)
        with open(self.__get_conf_path(), mode='w') as conf_file:
            yaml.dump(self.get_conf(), conf_file)

    def _load(self):
        if self._name not in list_dataset_names(self._datasets_root_path):
            raise DatasetNotFoundError('A dataset with name "{}" was not found in {}'
                                       .format(self._name, str(self._datasets_root_path)))
        train_df = pd.read_csv(self._train_path)
        if self._valid_path.exists():
            valid_df = pd.read_csv(self._valid_path)
        else:
            valid_df = None
        test_df = pd.read_csv(self._test_path)
        return train_df, valid_df, test_df

    def __load_conf(self):
        with open(self.__get_conf_path(), mode='r') as conf_file:
            return yaml.safe_load(conf_file)
    
    @staticmethod
    def __copy(df: pd.DataFrame):
        return df.copy(deep=True)

    def get_train(self):
        return DictionaryDataset.__copy(self._train_df)

    def get_valid(self):
        return DictionaryDataset.__copy(self._valid_df)

    def get_test(self):
        return DictionaryDataset.__copy(self._test_df)

    def get_all(self):
        if self._valid_df is None:
            return DictionaryDataset.__copy(pd.concat([self._train_df, self._test_df]))
        else:
            return DictionaryDataset.__copy(pd.concat([self._train_df, self._valid_df, self._test_df]))

    def get_conf(self):
        valid_path = str(self._valid_path) if self._valid_df is not None else None
        valid_count = len(self._valid_df) if self._valid_df is not None else None
        return {
            'name': self._name,
            'root_path': str(self._root_path),
            'train_path': str(self._train_path),
            'valid_path': valid_path,
            'test_path': str(self._test_path),
            'train_count': len(self._train_df),
            'valid_count': valid_count,
            'test_count': len(self._test_df),
            'source_file': self.__source_path
        }
