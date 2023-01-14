from pathlib import Path
from typing import List, Tuple

import pandas as pd

from common import ObjectWithConf, list_sorted_names_in_dir
from config import global_config, Config


def get_datasets_root_path() -> Path:
    return Path(global_config.storage.root) / global_config.storage.datasets_subdir


def list_dataset_names(datasets_root_path: Path = get_datasets_root_path()) -> List[str]:
    """
    Returns a list of dictionaries
    :param datasets_root_path: Optional override for datasets root directory
    :return: Names as string list
    """
    return list_sorted_names_in_dir(datasets_root_path)


class DatasetNotFoundError(FileNotFoundError):
    pass


class Dataset(ObjectWithConf):
    
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
        self._datasets_root_path = Path(datasets_root_path) \
            if datasets_root_path is not None else get_datasets_root_path()
        self._root_path, self._train_path, self._valid_path, self._test_path = self.__generate_paths()
        if train_df is not None and test_df is not None:
            self._train_df = train_df
            self._valid_df = valid_df
            self._test_df = test_df
            self.__source_path = source_path
            self._save(overwrite_if_exists)
        else:
            self._train_df, self._valid_df, self._test_df = self._load()
            self.__source_path = Path(self.__load_conf()['source_file'])

    def __generate_paths(self) -> Tuple[Path, Path, Path, Path]:
        root_path = self._datasets_root_path / self._name

        def get_file_path(set_type: str) -> Path:
            return root_path / f'{self._name}__{set_type}.csv'
        train_path = get_file_path('train')
        valid_path = get_file_path('valid')
        test_path = get_file_path('test')

        return root_path, train_path, valid_path, test_path

    def __get_conf_path(self):
        return self._root_path / f'{self._name}__conf.yaml'

    def _save(self, overwrite: bool):
        if not self._root_path.exists():
            self._root_path.mkdir(parents=True, exist_ok=overwrite)
        self._train_df.to_csv(self._train_path)
        if self._valid_df is not None:
            self._valid_df.to_csv(self._valid_path)
        self._test_df.to_csv(self._test_path)
        self._save_conf(self.__get_conf_path())

    def _load(self):
        if self._name not in list_dataset_names(self._datasets_root_path):
            raise DatasetNotFoundError(f'A dataset with name "{self._name}" was not found '
                                       f'in {str(self._datasets_root_path)}')
        train_df = pd.read_csv(self._train_path, index_col=0)
        if self._valid_path.exists():
            valid_df = pd.read_csv(self._valid_path, index_col=0)
        else:
            valid_df = None
        test_df = pd.read_csv(self._test_path, index_col=0)
        return train_df, valid_df, test_df

    def __load_conf(self):
        return self._load_conf(self.__get_conf_path())
    
    @staticmethod
    def __copy(df: pd.DataFrame):
        return df.copy(deep=True)

    def get_train(self):
        return Dataset.__copy(self._train_df)

    def get_valid(self):
        if self._valid_df is not None:
            return Dataset.__copy(self._valid_df)
        else:
            return None

    def get_test(self):
        return Dataset.__copy(self._test_df)

    def get_all(self):
        if self._valid_df is None:
            return Dataset.__copy(pd.concat([self._train_df, self._test_df]))
        else:
            return Dataset.__copy(pd.concat([self._train_df, self._valid_df, self._test_df]))

    def get_conf(self):
        valid_path = self._valid_path if self._valid_df is not None else None
        valid_count = len(self._valid_df) if self._valid_df is not None else None
        return Config({
            'name': self._name,
            'root_path': self._root_path,
            'train_path': self._train_path,
            'valid_path': valid_path,
            'test_path': self._test_path,
            'train_count': len(self._train_df),
            'valid_count': valid_count,
            'test_count': len(self._test_df),
            'source_file': self.__source_path,
            'datasets_root_path': self._datasets_root_path
        }).to_primitives_dict()
