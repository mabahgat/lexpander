from abc import abstractmethod
from pathlib import Path
from typing import List, Dict, Set, Optional, Type
import os.path
from liwc import Liwc
import pandas as pd

from common import ObjectWithConf
from config import global_config


class LexiconFormatError(ValueError):
    pass


class InvalidLabelError(ValueError):
    pass


class InvalidLexiconName(ValueError):
    pass


class LabelMapper(ObjectWithConf):
    """
    Maps and excludes labels.
    Label map is expected to map one child label to the another parent label.
    Parent labels are limited by the "Used Labels set"
    """

    def __init__(self, use_labels: Set[str], label_map: Dict[str, str]):
        """
        Creates a new instance
        :param use_labels: Set of labels
        :param label_map: Dictionary of labels mapped to others or None for top labels
        """
        if use_labels is None:
            raise ValueError('List of used labels is None')
        self.__use_labels = use_labels
        self.__label_map = label_map
        self.__cache = {}

    def map(self, label: str) -> str:
        if label in self.__cache:
            return self.__cache[label]
        else:
            new_label = None
            if label in self.__use_labels:
                new_label = label
            elif self.__label_map is not None:
                new_label = self.__search(label)
            self.__cache[label] = new_label
            return new_label

    def __search(self, label: str):
        if label not in self.__label_map:
            raise InvalidLabelError('Unexpected label "{}"'.format(label))
        new_label = self.__label_map[label]
        if new_label is None or new_label in self.__use_labels:
            return new_label
        else:
            return self.__search(new_label)

    def map_list(self, labels: List[str]) -> List[str]:
        mapped_labels = [self.map(label) for label in labels]
        mapped_labels = [label for label in mapped_labels if label is not None]
        return list(dict.fromkeys(mapped_labels))   # remove duplicates

    def get_conf(self):
        return {
            'used_labels': self.__use_labels,
            'label_mapping': self.__label_map
        }


class Lexicon(ObjectWithConf):

    @abstractmethod
    def label_term(self, token: str) -> List[str]:
        pass

    @abstractmethod
    def get_labels(self) -> Set[str]:
        """
        Get the set of all label classes
        :return: Unordered Set
        """
        pass


def get_lexicon(name: str = None, custom_lexicon_path: Path = None) -> Optional[Lexicon]:
    """
    Gets a lexicon instance by name. This only works for a specific predefined list of lexicons
    :param name: name string
    :param custom_lexicon_path: custom path to use while loading that lexicon
    :return: An instance of lexicon if a corresponding type is found
    """

    if name is None and custom_lexicon_path is None:
        raise ValueError('Either name or lexicon path has to be specified. Both are None')

    def get_instance(klass: Type[Lexicon]):
        if custom_lexicon_path is None:
            return klass()
        else:
            return klass(csv_path=custom_lexicon_path)

    if name is not None:
        if name == 'liwc2015':
            return Liwc2015(strict_matching=True)
        elif name == 'values':
            return get_instance(Values)
        elif name == 'liwc22':
            return get_instance(Liwc22)
        else:
            raise InvalidLexiconName(f'Invalid lexicon name "{name}"')
    else:
        return LookUpLexicon(custom_lexicon_path)


class LookUpLexicon(Lexicon):
    """
    Loads a lexicon based on exact match.
    """

    def __init__(self, file_path: str, sep: str = ','):
        self._path = file_path
        self._sep = sep
        self._lookup, self._labels = self.__load()

    def __load(self):
        if not os.path.exists(self._path):
            raise FileNotFoundError('Path invalid or does not exist "{}"'.format(self._path))

        lookup_dict = {}
        label_class_set = set()
        with open(self._path, mode='r', encoding='utf8') as lexicon_file:
            for line_index, line in enumerate(lexicon_file.readlines()):
                line = line.strip()
                parts = line.split(sep=self._sep)
                if len(parts) != 2:
                    raise LexiconFormatError('Unexpected entry in lexicon at line {}: "{}"'.format(line_index, line))
                word, label = parts
                if word not in lookup_dict:
                    lookup_dict[word] = []
                lookup_dict[word].append(label)
                label_class_set.add(label)
            return lookup_dict, label_class_set

    def label_term(self, token: str) -> List[str]:
        if token not in self._lookup:
            return []
        else:
            return self._lookup[token]

    def get_labels(self) -> Set[str]:
        return self._labels

    def get_conf(self) -> Dict[str, object]:
        return {
            'file_path': self._path,
            'sep': self._sep,
            'labels': self._labels,
            'label_count': len(self._labels),
            'entries_count': len(self._lookup)
        }


class Liwc2015(Lexicon):

    DEFAULT_LABELS_MAP = {
        'function': None,
        'pronoun': 'function',
        'ppron': 'pronoun',
        'i': 'ppron',
        'we': 'ppron',
        'you': 'ppron',
        'shehe': 'ppron',
        'they': 'ppron',
        'ipron': 'pronoun',
        'article': 'function',
        'prep': 'function',
        'auxverb': 'function',
        'adverb': 'function',
        'conj': 'function',
        'negate': 'function',
        'verb': 'function',
        'adj': 'function',
        'compare': 'function',
        'interrog': 'function',
        'number': 'function',
        'quant': 'function',
        'affect': None,
        'posemo': 'affect',
        'negemo': 'affect',
        'anx': 'negemo',
        'anger': 'negemo',
        'sad': 'negemo',
        'social': None,
        'family': 'social',
        'friend': 'social',
        'female': 'social',
        'male': 'social',
        'cogproc': None,
        'insight': 'cogproc',
        'cause': 'cogproc',
        'discrep': 'cogproc',
        'tentat': 'cogproc',
        'certain': 'cogproc',
        'differ': 'cogproc',
        'percept': None,
        'see': 'percept',
        'hear': 'percept',
        'feel': 'percept',
        'bio': None,
        'body': 'bio',
        'health': 'bio',
        'sexual': 'bio',
        'ingest': 'bio',
        'drives': None,
        'affiliation': 'drives',
        'achiev': 'drives',
        'power': 'drives',
        'reward': 'drives',
        'risk': 'drives',
        'timeorient': None,
        'focuspast': 'timeorient',
        'focuspresent': 'timeorient',
        'focusfuture': 'timeorient',
        'relativ': None,
        'motion': 'relativ',
        'space': 'relativ',
        'time': 'relativ',
        'pconcern': None,
        'work': 'pconcern',
        'leisure': 'pconcern',
        'home': 'pconcern',
        'money': 'pconcern',
        'relig': 'pconcern',
        'death': 'pconcern',
        'informal': None,
        'swear': 'informal',
        'netspeak': 'informal',
        'assent': 'informal',
        'nonflu': 'informal',
        'filler': 'informal'
    }

    def __init__(self,
                 use_labels: Set[str] = None,
                 label_map: Dict[str, str] = DEFAULT_LABELS_MAP,
                 dic_path: str = None,
                 strict_matching: bool = False):
        self.__dic_path = dic_path if dic_path is not None else global_config.lexicons.liwc2015.dic
        self.__strict_matching = strict_matching
        self.__liwc = self.__load()
        self.__liwc_lookup = self.__load_strict() if self.__strict_matching else None

        if use_labels is not None:
            self.__label_mapper = LabelMapper(use_labels=use_labels, label_map=label_map)
            self.__labels = use_labels
        else:
            self.__label_mapper = None
            self.__labels = set(self.__liwc.categories.values())

    def __load(self):
        return Liwc(self.__dic_path)

    def __load_strict(self):
        def remove_wild_card(entry_str):
            if entry_str.endswith('*'):
                return entry_str[0:-1]
            return entry_str

        return {remove_wild_card(key): value for key, value in self.__liwc.lexicon.items()}

    def label_term(self, token: str) -> List[str]:
        if self.__strict_matching:
            labels = self.__liwc_lookup.get(token, [])
        else:
            labels = self.__liwc.search(token)
        if self.__label_mapper is not None:
            return self.__label_mapper.map_list(labels)
        else:
            return labels

    def get_labels(self) -> Set[str]:
        return self.__labels

    def get_conf(self) -> Dict[str, object]:
        return {
            'file_path': self.__dic_path,
            'strict': self.__strict_matching,
            'labels': self.__labels,
            'label_count': len(self.__labels),
            'label_map': self.__label_mapper,
            'name': 'liwc2015'
        }


class LookUpLexiconWithMapping(LookUpLexicon):

    def __init__(self,
                 use_labels: Set[str],
                 label_map: Dict[str, str],
                 csv_path: str):
        super().__init__(file_path=csv_path)
        if use_labels is not None:
            self.__label_mapper = LabelMapper(use_labels=use_labels, label_map=label_map)
            self._labels = use_labels
        else:
            self.__label_mapper = None

    def label_term(self, token: str) -> List[str]:
        labels = super().label_term(token)
        if self.__label_mapper is not None:
            return self.__label_mapper.map_list(labels)
        else:
            return labels

    def get_conf(self) -> Dict[str, object]:
        conf = super().get_conf()
        if self.__label_mapper is not None:
            conf['label_map'] = self.__label_mapper.get_conf()
        else:
            conf['label_map'] = 'None'
        return conf


class Values(LookUpLexiconWithMapping):

    DEFAULT_LABEL_MAP = {
        'autonomy': 'life',
        'creativity': 'cognition',
        'emotion': 'cognition',
        'moral': 'cognition',
        'cognition': 'life',
        'future': 'cognition',
        'thinking': 'cognition',
        'security': 'order',
        'inner-peace': 'order',
        'order': 'life',
        'justice': 'life',
        'advice': 'life',
        'career': 'life',
        'achievement': 'life',
        'wealth': 'life',
        'health': 'life',
        'learning': 'life',
        'nature': 'life',
        'animals': 'life',
        'purpose': 'work-ethic',
        'responsible': 'work-ethic',
        'hard-work': 'work-ethic',
        'work-ethic': None,
        'perseverance': 'work-ethic',
        'feeling-good': None,
        'forgiving': 'accepting-others',
        'accepting-others': None,
        'helping-others': 'society',
        'gratitude': None,
        'dedication': None,
        'self-confidence': None,
        'optimisim': None,
        'honesty': 'truth',
        'truth': None,
        'spirituality': 'religion',
        'religion': None,
        'significant-other': 'relationships',
        'marriage': 'significant-other',
        'friends': 'relationships',
        'relationships': 'social',
        'family': 'relationships',
        'parents': 'family',
        'siblings': 'family',
        'social': None,
        'children': 'family',
        'society': 'social',
        'art': 'life',
        'respect': 'self-confidence',
        'life': None
    }

    def __init__(self,
                 use_labels: Set[str] = None,
                 label_map: Dict[str, str] = DEFAULT_LABEL_MAP,
                 csv_path: str = None):
        if csv_path is None:
            csv_path = global_config.lexicons.values.csv
        super().__init__(use_labels=use_labels, label_map=label_map, csv_path=csv_path)

    def get_conf(self) -> Dict[str, object]:
        conf = super().get_conf()
        conf['name'] = 'values'
        return conf


class Liwc22(Lexicon):
    """
    Liwc 22 lexicon.
    The loaded file should contain all labels corresponding to each word.
    That means there's no need for mapping.
    """

    EXCLUDED_COLUMNS = [
        'ColumnID',
        'Segment',
        'WC',
        'Analytic',
        'Clout',
        'Authentic',
        'Tone',
        'WPS',
        'BigWords',
        'Dic',
        'Linguistic',
        'function',
        'pronoun',
        'ppron',
        'i',
        'we',
        'you',
        'shehe',
        'they',
        'ipron',
        'det',
        'article',
        'number',
        'prep',
        'auxverb',
        'adverb',
        'conj',
        'negate',
        'verb',
        'adj',
        'focuspast',
        'focuspresent',
        'focusfuture',
        'AllPunc',
        'Period',
        'Comma',
        'QMark',
        'Exclam',
        'Apostro',
        'OtherP'
    ]

    def __init__(self, csv_path: str = None,
                 use_labels: Set[str] = None,
                 from_tool_output: bool = False):
        """
        Create a new instance from file
        :param csv_path: optional path string, otherwise value from configuration taken
        :param use_labels: values for used labels only
        :param from_tool_output: if the passed csv file is usual pipe separated csv or generated from LIWC22 tool output
        """
        self.__csv_path = csv_path
        self.__use_labels = use_labels
        self.__from_tool_output = from_tool_output
        self._lookup = self.__load()

    def __load(self):
        if self.__from_tool_output:
            return self.__load_from_tool_output()
        else:
            return self.__load_from_csv()

    def __load_from_tool_output(self):
        if self.__csv_path is None:
            self.__csv_path = global_config.lexicons.liwc22.tool_out
        csv_df = pd.read_csv(self.__csv_path, index_col='Text')
        csv_df = csv_df[csv_df['Dic'] > 0]
        csv_df = csv_df.drop(columns=Liwc22.EXCLUDED_COLUMNS)
        csv_df = csv_df.drop(columns=csv_df.columns[0])
        i = 0

        def get_annotations(row_dict):
            nonlocal i
            i += 1
            labels_lst = []
            for cat_str, value in row_dict.items():
                if isinstance(value, str):
                    raise LexiconFormatError(
                        'Error for word "{}" at line "{}" in file {}'.format(value, i, self.__csv_path))
                if value > 0:
                    labels_lst.append(cat_str)
            if self.__use_labels:
                labels_lst = [label for label in labels_lst if label in self.__use_labels]
            return labels_lst

        csv_df['labels'] = csv_df.apply(get_annotations, axis=1)
        return csv_df['labels'].to_dict()

    def __load_from_csv(self):
        if self.__csv_path is None:
            self.__csv_path = global_config.lexicons.liwc22.csv
        csv_df = pd.read_csv(self.__csv_path, index_col=0, names=['word', 'labels_raw'])

        def to_label_list(raw_labels: str) -> List[str]:
            labels = raw_labels.split('|')
            if self.__use_labels:
                labels = [label for label in labels if label in self.__use_labels]
            return labels
        csv_df['labels'] = csv_df['labels_raw'].apply(to_label_list)
        return csv_df['labels'].to_dict()

    def label_term(self, token: str) -> List[str]:
        token = token.lower()
        if token not in self._lookup:
            return []
        else:
            return self._lookup[token]

    def get_labels(self) -> Set[str]:
        if self.__use_labels is not None:
            return self.__use_labels
        else:
            labels = set()
            for token_labels in self._lookup.values():
                for label in token_labels:
                    labels.add(label)
            return labels

    def get_conf(self) -> Dict[str, object]:
        labels = self.get_labels()
        return {
            'name': 'liwc22',
            'file_path': self.__csv_path,
            'labels': labels,
            'labels_count': len(labels),
            'from_tool_output': self.__from_tool_output
        }
