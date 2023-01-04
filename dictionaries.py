import re
import urllib.parse
from abc import abstractmethod
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
        self._file_type = file_type
        if file_path is None:
            file_path = self._use_default_path(self._file_type)
        super().__init__(name, file_path)

    def _use_default_path(self, dict_conf_name: str):
        """
        Returns the default path for a dictionary
        :param dict_conf_name: Name string as in the yaml configuration file
        :return:
        """
        if self._file_type == DictionaryWithTwoFormats._RAW_FILE_TYPE:
            return global_config.dictionaries[dict_conf_name].csv
        elif self._file_type == DictionaryWithTwoFormats._CSV_FILE_TYPE:
            return global_config.dictionaries[dict_conf_name].raw
        else:
            raise BadFileType('Unknown file type "{}"'.format(self._file_type))

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
        return pd.read_csv(self._file_type, index_col=0)

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
        super().__init__(name='UrbanDictionary', file_type=file_type, file_path=file_path)

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
        super().__init__(name='Wiktionary', file_type=file_type, file_path=file_path)

    def _load_raw(self) -> pd.DataFrame:
        df = pd.read_csv(self._file_path, index_col=0)
        df.word = df.word.str.lower()
        df.meaning = df.meaning.fillna('')
        df.example = df.example.fillna('')
        df[Dictionary.TEXT_COLUMN] = df.apply(lambda r: '{} {}'.format(r['meaning'], r['example']).strip(), axis=1)
        df[Dictionary.QUALITY_COLUMN] = 100     # wiktionary entries are reviewed
        return df[[Dictionary.WORD_COLUMN, Dictionary.TEXT_COLUMN, Dictionary.QUALITY_COLUMN]]
