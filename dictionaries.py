import re
import urllib.parse
from abc import abstractmethod
from typing import Type

import pandas as pd

from common import ObjectWithConf
from config import global_config


class Dictionary(ObjectWithConf):

    WORD_COLUMN = 'word'
    TEXT_COLUMN = 'text'
    QUALITY_COLUMN = 'quality'

    def __init__(self, name: str, file_path: str):
        self._name = name
        self._file_path = file_path
        self._records = self._load()

    @abstractmethod
    def _load(self) -> pd.DataFrame:
        pass

    def get_all_records(self) -> pd.DataFrame:
        """
        Returns all records with index as record ID, word, text, quality columns.
        :return: Pandas Dataframe
        """
        return self._records.copy(deep=True)

    def get_definition(self, term: str) -> str:
        """
        Returns the definitions for a single term
        :param term:
        :return:
        """
        return self._records.loc[term]

    def get_conf(self):
        return {
            'name': self._name,
            'file_path': self._file_path,
            'count': len(self._records)
        }


class SimpleDictionary(Dictionary):

    def _load(self) -> pd.DataFrame:
        return pd.read_csv(self._file_path, index_col=0)


class BadFileType(TypeError):
    pass


class ColumnNotFound(ValueError):
    """
    Used when a column is missing from a dictionary, and it was expected to exist.
    Example, if a dictionary does not have a "quality" column, but it is expected to contain it somewhere in the code
    """
    pass


def get_dictionary(name: str, **kwargs) -> Dictionary:
    """
    Gets a dictionary by name. This only works for specific predefined list of dictionaries
    :param name: name string
    :param kwargs: parameters passed on to dictionary initialization
    :return: An instance of dictionary if corresponding type is found
    """
    def get_instance(klass: Type[Dictionary]):
        return klass(**kwargs)
    if name == 'wiktionary':
        return get_instance(Wiktionary)
    elif name == 'ud' or name == 'urban_dictionary':
        return get_instance(UrbanDictionary)
    elif name == 'opted':
        return get_instance(Opted)
    else:
        params = kwargs.copy()
        params['name'] = name
        return SimpleDictionary(**params)


class DictionaryWithTwoFormats(Dictionary):
    """
    Parse raw file and csv formats.
    Raw files are custom formatted files
    CSV format is expected to have: ID (used as index), word, text and quality columns.
    """

    _RAW_FILE_TYPE = 'raw'
    _CSV_FILE_TYPE = 'csv'

    def __init__(self,
                 name: str,
                 file_type: str = _CSV_FILE_TYPE,
                 file_path: str = None):
        if file_type not in [DictionaryWithTwoFormats._RAW_FILE_TYPE, DictionaryWithTwoFormats._CSV_FILE_TYPE]:
            raise BadFileType(f'Unknown file type "{file_type}"')

        self._file_type = file_type
        if file_path is None:
            file_path = self._use_default_path(name, file_type)
        super().__init__(name, file_path)

    def _use_default_path(self, dict_name: str, file_type: str):
        """
        Returns the default path for a dictionary
        :param dict_name: Name string as in the yaml configuration file
        :param file_type: type name string
        :return:
        """
        return global_config.dictionaries[dict_name][file_type]

    def _load(self) -> pd.DataFrame:
        if self._file_type == DictionaryWithTwoFormats._RAW_FILE_TYPE:
            return self._load_raw()
        elif self._file_type == DictionaryWithTwoFormats._CSV_FILE_TYPE:
            return self._load_csv()
        else:
            raise TypeError('Unexpected type {}'.format(self._file_type))

    @abstractmethod
    def _load_raw(self) -> pd.DataFrame:
        pass

    def _load_csv(self):
        return pd.read_csv(self._file_path, index_col=0)

    def save_as_csv(self, file_path: str):
        self._records.to_csv(file_path)

    def get_conf(self):
        conf = super().get_conf()
        conf['file_type'] = self._file_type
        return conf


class UrbanDictionary(DictionaryWithTwoFormats):
    """
    Raw files are expected to be custom format with pipe separated columns.
    Text column is generated from appending tags (lower cased and '#' are replaced by space), text and quality.
    """

    def __init__(self, file_path: str = None, file_type: str = 'csv'):
        super().__init__(name='urban_dictionary', file_type=file_type, file_path=file_path)

    def _load_raw(self) -> pd.DataFrame:
        df = pd.read_csv(self._file_path, sep=r'\|')
        UrbanDictionary.__fix_text_column_inplace(df, 'word', do_lower=True)
        UrbanDictionary.__fix_text_column_inplace(df, 'meaning')
        UrbanDictionary.__fix_text_column_inplace(df, 'example')
        df = UrbanDictionary.__remove_bad_entries(df)

        def generate_text(row) -> str:
            tags = str(row['tagList']).replace('#', ' ').lower()
            return '{} {} {}'.format(tags, str(row['meaning']), str(row['example'])).strip()
        df[Dictionary.TEXT_COLUMN] = df.apply(generate_text, axis=1)
        df[Dictionary.QUALITY_COLUMN] = pd.to_numeric(df.numLikes) - pd.to_numeric(df.numDislikes)
        df = df[[Dictionary.WORD_COLUMN, Dictionary.TEXT_COLUMN, Dictionary.QUALITY_COLUMN]]
        df.index.name = 'id'
        return df

    @staticmethod
    def __decode_text(text_str: str) -> str:
        return urllib.parse.unquote_plus(text_str)

    @staticmethod
    def __fix_text_column_inplace(df: pd.DataFrame,
                                  column_name: str,
                                  do_lower: bool = False) -> None:
        df[column_name].fillna('', inplace=True)
        df[column_name] = df[column_name].apply(UrbanDictionary.__decode_text)
        if do_lower:
            df[column_name] = df[column_name].str.lower()

    @staticmethod
    def __remove_bad_entries(df: pd.DataFrame) -> pd.DataFrame:
        df = df[df['numLikes'].apply(lambda v: re.match(r'\d+', str(v)) is not None)]
        df = df[df['numDislikes'].apply(lambda v: re.match(r'\d+', str(v)) is not None)]
        return df


class Wiktionary(DictionaryWithTwoFormats):

    def __init__(self, file_path: str = None, file_type: str = 'csv'):
        super().__init__(name='wiktionary', file_type=file_type, file_path=file_path)

    def _load_raw(self) -> pd.DataFrame:
        df = pd.read_csv(self._file_path, index_col=0)
        df.word = df.word.str.lower()
        df.meaning = df.meaning.fillna('')
        df.example = df.example.fillna('')
        df[Dictionary.TEXT_COLUMN] = df.apply(lambda r: '{} {}'.format(r['meaning'], r['example']).strip(), axis=1)
        df[Dictionary.QUALITY_COLUMN] = 100     # wiktionary entries are reviewed
        return df[[Dictionary.WORD_COLUMN, Dictionary.TEXT_COLUMN, Dictionary.QUALITY_COLUMN]]


class Opted(SimpleDictionary):

    def __init__(self, file_path: str = None):
        if file_path is None:
            file_path = global_config.dictionaries['opted']['csv']
        super().__init__(name='opted', file_path=file_path)

    def _load(self) -> pd.DataFrame:
        df = super()._load()
        df.index = df.index.str.lower()
        df.index.rename('lookup_index', inplace=True)   # TODO fix index for Dictionary class
        df.text = df.text.fillna('')    # TODO add this for all dictionaries, breaks bert tokenization if NaN exists
        df[Dictionary.WORD_COLUMN] = df.index
        assert Dictionary.WORD_COLUMN in df.columns
        assert Dictionary.TEXT_COLUMN in df.columns
        return df
