import logging
from pathlib import Path
from random import Random
from typing import List, Union, Dict, Set
import functools

import pandas as pd
from tqdm import tqdm

from common import ObjectWithConf
from datasets.base import Dataset
from dictionaries import Dictionary
from lexicons import Lexicon
from lists import LookUpList, NamesLookUp, StopWordsLookUp, FileLookUp


class DatasetGenerator(ObjectWithConf):
	"""
	Generates a dataset.
	If multiple dictionaries are passed then the same test set will be generated for each dictionary.
	Same test set means same list of terms are chosen for datasets corresponding to the multiple dictionaries.
	Same train set can be optionally generated.
	Terms selected for the training set in the datasets will be the same for the corresponding datasets.
	To keep the same balance in the overall data for multiple dictionary case,
	the average of the total instances for each label is taken across the multiple dictionaries.
	"""

	LABEL_COLUMN = 'label'

	def __init__(self,
				 exp_name: str,
				 lexicon: Lexicon,
				 dictionaries: Union[List[Dictionary], Dictionary],
				 test_count: int = None,
				 force_test_count: bool = False,
				 test_percentage: float = None,
				 same_train_set: bool = False,
				 top_quality_count: int = None,
				 quality_threshold: int = None,
				 exclusions: List[Union[LookUpList, str, Path]] = None,
				 overwrite_if_exists: bool = False,
				 dataset_root_path: Path = None):
		"""
		New instance of Dataset Generator
		:param exp_name: Experiment name string. Should be meaningful to easily identify different experiments
		:param lexicon: Lexicon instance to annotate the dictionaries with
		:param dictionaries: a dictionary instance or a list of dictionaries
		:param test_count: suggested number of test instances. The generator will attempt to satisfy the number.
		See force_test_count
		:param force_test_count: If true, if actual test count less than select test count,
		more test instance will be added. Resulting test set will be slightly imbalanced
		:param test_percentage: Alternative for test_count, float value for percentage
		:param same_train_set: whether to use same terms or not
		:param top_quality_count: Optional to select the top-n terms per word
		:param quality_threshold: Optional threshold (inclusive) to discard instances with lower quality value
		:param exclusions: List of lists of terms to be excluded from the dataset.
			None will use the default list which excludes names and stopwords.
			An empty list will skip all exclusions
		:param overwrite_if_exists: Passed on to Datasets class to either overwrite or throw error if dataset exists
		:param dataset_root_path: Overrides default path for storing datasets
		"""
		if test_count is None and test_percentage is None:
			raise ValueError('Both test_count and test_percentage can not be none')
		if test_count is not None and test_percentage is not None:
			raise ValueError('Either test_count or test_percentage can be defined not both')

		self.__logger = logging.getLogger(__name__)

		self.__exp_name = exp_name
		self.__lexicon = lexicon
		self.__dictionaries = dictionaries if type(dictionaries) is list else [dictionaries]
		self.__user_test_count = test_count
		self.__actual_test_count = None
		self.__force_test_count = force_test_count
		self.__test_percentage = test_percentage
		self.__same_train_set = same_train_set
		self.__top_quality_count = top_quality_count
		self.__quality_threshold = quality_threshold
		self.__exclusions = DatasetGenerator.__get_exclusions(exclusions)
		self.__overwrite_if_exists = overwrite_if_exists
		self.__dataset_root_path = dataset_root_path

		self.__random = Random(0)

	@staticmethod
	def __get_exclusions(exclusions: List[Union[LookUpList, str, Path]]) -> List[LookUpList]:
		if exclusions is None:
			return [NamesLookUp(), StopWordsLookUp()]
		else:
			def get_lookup_by_type(v):
				if type(v) is LookUpList:
					return v
				elif type(v) is str:
					return FileLookUp(Path(v))
				elif type(v) is Path:
					return FileLookUp(v)
				else:
					raise TypeError(f'Unexpected type for exclusion list "{type(v)}"')
			return [get_lookup_by_type(v) for v in exclusions]

	def generate(self) -> List[Dataset]:
		labeled_dictionaries = [entries.sample(frac=1, random_state=0) for entries in self.__get_labeled_entries()]
		test_terms = self.__generate_test_terms_for_all(labeled_dictionaries)
		test_sets = self.__select_test_sets(labeled_dictionaries, test_terms)
		train_sets = DatasetGenerator.__select_train_sets(labeled_dictionaries, test_terms)

		assert len(self.__dictionaries) == len(test_sets)
		assert len(self.__dictionaries) == len(train_sets)

		return [Dataset(name=f'{self.__exp_name}_{dictionary.get_conf()["name"]}',
						train_df=train,
						test_df=test,
						overwrite_if_exists=self.__overwrite_if_exists,
						source_path=dictionary.get_conf()['file_path'],
						datasets_root_path=self.__dataset_root_path)
				for train, test, dictionary in zip(train_sets, test_sets, self.__dictionaries)]

	def __get_labeled_entries(self) -> List[pd.DataFrame]:
		"""
		Labels each dictionary, returns only labeled entries with the required minimum quality if quality is defined.
		For multiple labels, only the first label is selected
		:return: List of dataframes one for each dictionary
		"""
		return [self.__get_dictionary_valid_entries(dictionary) for dictionary in self.__dictionaries]

	def __get_dictionary_valid_entries(self, dictionary: Dictionary) -> pd.DataFrame:
		df = dictionary.get_all_records()
		self.__logger.info(f'Dictionary: {dictionary.get_conf()["name"]} - all entry count: {len(df)}')

		tqdm.pandas(desc=f'Labeling {dictionary.get_conf()["name"]}')
		df[DatasetGenerator.LABEL_COLUMN] = df.word.progress_apply(lambda term: self.__lexicon.label_term(term))

		df = df[df.label.apply(lambda l: len(l) > 0)]
		df.label = df.label.apply(lambda l: l[0])
		self.__logger.info(
			f'Dictionary: {dictionary.get_conf()["name"]} - after removing entries not in lexicon: {len(df)}')

		def contained_in_exclusions(term: str) -> bool:
			for exclusion in self.__exclusions:
				if exclusion.contains(term):
					return True
			return False
		self.__logger.info(f'Applying {len(self.__exclusions)} exclusions')
		df = df[~df.word.progress_apply(contained_in_exclusions)]
		self.__logger.info(f'Dictionary: {dictionary.get_conf()["name"]} - after applying exclusions: {len(df)}')

		if self.__top_quality_count is not None and Dictionary.QUALITY_COLUMN in df:
			self.__logger.info(f'Selecting top {self.__top_quality_count} records for each term')
			df = df.groupby(by=Dictionary.WORD_COLUMN, as_index=False).head(self.__top_quality_count)
			self.__logger.info(
				f'Dictionary: {dictionary.get_conf()["name"]} - '
				f'after selection top {self.__top_quality_count}: {len(df)}')

		if self.__quality_threshold is not None and Dictionary.QUALITY_COLUMN in df:
			self.__logger.info(f'Filtering records with threshold {self.__quality_threshold} inclusive')
			df = df[df.quality >= self.__quality_threshold]
			self.__logger.info(
				f'Dictionary: {dictionary.get_conf()["name"]} - '
				f'after filtering with threshold (inclusive) {self.__quality_threshold}: {len(df)}')
		return df

	@staticmethod
	def __compute_label_distribution(labeled_frames: List[pd.DataFrame]) -> Dict[str, float]:
		counts = [DatasetGenerator.__compute_label_counts_for_dictionary(frame) for frame in labeled_frames]
		sum_per_label = pd.DataFrame.from_dict(counts).sum()
		total_terms = sum_per_label.sum()
		return (sum_per_label / total_terms).to_dict()

	@staticmethod
	def __compute_label_counts_for_dictionary(df: pd.DataFrame) -> Dict[str, int]:
		return df.groupby(by=DatasetGenerator.LABEL_COLUMN).size().to_dict()

	def __generate_test_terms_for_all(self, labeled_dictionaries: List[pd.DataFrame]) -> Set[str]:
		"""
		Generates the list of terms to be used for the test set across the datasets generated from the different
		dictionaries
		:param labeled_dictionaries: dataframes for the content in the dictionaries labeled with the lexicon
		:return:
		"""
		label_distribution = DatasetGenerator.__compute_label_distribution(labeled_dictionaries)
		available_terms = DatasetGenerator.__get_common_terms(labeled_dictionaries)
		average_set_size = sum([len(dictionary) for dictionary in labeled_dictionaries]) / len(labeled_dictionaries)
		return self.__select_test_terms(available_terms, label_distribution, average_set_size)

	def __select_test_sets(self, labeled_dictionaries: List[pd.DataFrame], test_terms: Set[str]) -> List[pd.DataFrame]:
		"""
		Generates a test set for each dictionary
		:param labeled_dictionaries: data frames for dictionary content with labels from lexicon
		:param test_terms: List of terms to be used
		:return: list of dataframes each dataframe is the test set corresponding to the passed labeled dictionary in order
		"""
		return [self.__generate_test_set(dictionary_df, test_terms) for dictionary_df in labeled_dictionaries]

	def __select_test_terms(self,
							available_terms: Set[str],
							label_distribution: Dict[str, float],
							average_set_size: float) -> Set[str]:
		"""
		Selects test terms that will be used for all dictionaries
		:param available_terms:
		:param label_distribution:
		:param average_set_size:
		:return: Set of terms
		"""
		target_test_size = self.__user_test_count if self.__user_test_count is not None else round(
			self.__test_percentage * average_set_size)
		label_set_sizes = {label: round(target_test_size * percentage) for label, percentage in
						   label_distribution.items()}
		actual_size = sum(label_set_sizes.values())

		if actual_size < target_test_size and self.__force_test_count:
			remaining_terms_count = target_test_size - actual_size
			add_to_labels = self.__random.sample(label_set_sizes.keys(), remaining_terms_count)
			for label in add_to_labels:
				label_set_sizes[label] += 1
			actual_size = sum(label_set_sizes.values())

		self.__actual_test_count = actual_size

		labeled_terms = {term: self.__lexicon.label_term(term)[0] for term in available_terms}
		terms_per_label = {}
		for term, label in labeled_terms.items():
			if label not in terms_per_label:
				terms_per_label[label] = []
			terms_per_label[label].append(term)

		selected_terms = set()
		for label, term_list in terms_per_label.items():
			count_to_take = label_set_sizes[label]
			self.__random.shuffle(term_list)
			label_selected_terms = term_list[:count_to_take]
			selected_terms |= set(label_selected_terms)

		assert len(selected_terms) == self.__actual_test_count

		return selected_terms

	def __generate_test_set(self, dictionary_df: pd.DataFrame, test_terms: Set[str]):
		df = dictionary_df[dictionary_df.word.isin(test_terms)]
		df = df.sample(frac=1, random_state=0)
		if Dictionary.QUALITY_COLUMN in df:
			df = df\
				.groupby(by=Dictionary.WORD_COLUMN, as_index=False)\
				.apply(lambda g: g.sort_values(by=Dictionary.QUALITY_COLUMN, ascending=False).head(1))
		else:
			df = df.groupby(by=Dictionary.WORD_COLUMN, as_index=False).head(1)

		assert len(df) == self.__actual_test_count

		return df

	@staticmethod
	def __get_common_terms(dictionaries: List[pd.DataFrame]) -> Set[str]:
		term_sets = [set(df.word.to_list()) for df in dictionaries]
		return functools.reduce(lambda s1, s2: s1 & s2, term_sets)

	@staticmethod
	def __select_train_sets(labeled_dictionaries: List[pd.DataFrame], test_terms: Set[str]) -> List[pd.DataFrame]:
		return [df[~df.word.isin(test_terms)] for df in labeled_dictionaries]

	def get_conf(self):
		exclusions = None if self.__exclusions is None else [e.get_conf() for e in self.__exclusions]
		return {
			'name': self.__exp_name,
			'lexicon': self.__lexicon.get_conf(),
			'dictionaries': [d.get_conf() for d in self.__dictionaries],
			'user_test_count': self.__user_test_count,
			'actual_test_count': self.__actual_test_count,
			'force_test_count': self.__force_test_count,
			'test_percentage': self.__test_percentage,
			'same_train_set': self.__same_train_set,
			'exclusions': exclusions,
			'overwrite_if_exists': self.__overwrite_if_exists,
			'datasets_root_path': self.__dataset_root_path
		}
