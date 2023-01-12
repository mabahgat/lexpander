from abc import ABC
from pathlib import Path

from nltk.corpus import stopwords
import nltk

from common import ObjectWithConf
from config import global_config


class LookUpList(ObjectWithConf, ABC):

    def __init__(self):
        self._lookup = set()

    def contains(self, value: str):
        return value.lower() in self._lookup


class FileLookUp(LookUpList):

    def __init__(self, path: Path):
        super().__init__()
        self.__list_path = path
        self._lookup = self.__load()

    def __load(self):
        lookup = set()
        with open(self.__list_path, mode='r', encoding='utf8') as list_file:
            for line in list_file.readlines():
                line = line.strip()
                if line:
                    lookup.add(line.lower())
        return lookup

    def get_conf(self):
        return {
            'file_path': str(self.__list_path),
            'count': len(self._lookup),
            'case_sensitive': False
        }


class NamesLookUp(FileLookUp):

    def __init__(self, path: Path = None):
        list_path = path if path is not None else global_config.lists.names.lst
        super().__init__(path=list_path)

    def get_conf(self):
        conf = super().get_conf()
        conf['name'] = 'names'
        return conf


class StopWordsLookUp(LookUpList):

    def __init__(self):
        super().__init__()
        self._lookup = StopWordsLookUp.__load()

    @staticmethod
    def __load():
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        return set(stopwords.words('english'))

    def contains(self, value: str):
        return value.lower() in self._lookup

    def get_conf(self):
        return {
            'library': 'nltk {}'.format(nltk.__version__),
            'case_sensitive': False,
            'count': len(self._lookup)
        }
