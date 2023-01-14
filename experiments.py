import logging
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd

from common import ObjectWithConf, list_sorted_names_in_dir
from config import global_config, Config
from datasets.base import Dataset
from datasets.generators import DatasetGenerator
from dictionaries import get_dictionary, Dictionary
from lexicons import get_lexicon, ExpandedLexicon
from models.base import Model
from models.utils import get_model_by_name


def get_experiments_root_path() -> Path:
	return Path(global_config.storage.root) / global_config.storage.experiments


def list_experiment_names(experiments_root_path: Path = get_experiments_root_path()) -> List[str]:
	"""
	Return list of stored experiments
	:param experiments_root_path:
	:return:
	"""
	return list_sorted_names_in_dir(experiments_root_path)


class Experiment(ObjectWithConf):
	"""
	End to end running of an experiment:
	a. generates the datasets
	b. train the model
	c. evaluate the models
	d. apply the model to the whole dictionary
	c. saves the output
	The instance is initialized with a configuration dictionary.
	If optional values are not specified then defaults are used.
	"""

	def __init__(self, conf: Dict[str, object] = None, conf_path: Path = None):
		"""
		Gets an instance of experiment object
		:param conf: dictionary with experiment configuration
		:param conf_path: Path containing experiment configuration
		"""
		if conf is None and conf_path is None:
			raise ValueError('Either configuration dict or configuration path has to be specified')
		if conf is not None and conf_path is not None:
			raise ValueError('Both configuration dictionary and configuration path can not be specified together')

		self.__logger = logging.getLogger(__name__)

		config_dict = ObjectWithConf._load_conf(conf_path) if conf_path is not None else conf
		self.__conf = Config(config_dict)

		# Reminder: Any new parameters need to be added to both constructor and __load
		self.__exp_name = self.__conf.exp_name
		self.__experiments_root_paths = Path(self.__conf.experiments_root_path) \
			if 'experiments_root_path' in self.__conf else get_experiments_root_path()

		self.__do_label_dictionaries = self.__conf.do_label_dictionaries \
			if 'do_label_dictionaries' in self.__conf else False  # default False: it takes hours to label a dictionary
		self._labeled_output_path = Path(self.__conf.labeled_output_path) \
			if 'labeled_output_path' in self.__conf else None
		self.__expanded_root_path = Path(self.__conf.expanded_root_path) \
			if 'expanded_root_path' in self.__conf else None
		self.__overwrite_if_exists = self.__conf.overwrite_if_exists \
			if 'overwrite_if_exists' in self.__conf else False

		self.__label_prob_threshold = self.__conf.label_prob_threshold \
			if 'label_prob_threshold' in self.__conf else 0

		if self.__exists():
			self.__load()
			self.__loaded_experiment = True
		else:
			self.__lexicon = None
			self.__dictionaries = None
			self.__datasets = None
			self.__models = None
			self.__results = None
			self.__labeled_dictionary_dfs = None
			self.__expanded_lexicons = None
			self.__loaded_experiment = False

	def __exists(self):
		return self.__exp_name in list_experiment_names(self.__experiments_root_paths)

	def __load(self):
		self.__logger.info(f'Loading experiment {self.__exp_name}')
		self.__conf = Config(ObjectWithConf._load_conf(self.__get_conf_path()))  # has results extra, rest is identical

		self.__logger.info('Loading lexicon')
		self.__lexicon = self.__get_lexicon()
		self.__logger.info(f'Lexicon {type(self.__lexicon)} loaded')

		self.__logger.info('Loading dictionaries')
		self.__dictionaries = self.__get_dictionaries()
		self.__logger.info(
			f'Loaded {len(self.__dictionaries)} '
			f'dictionaries: {[d.get_conf()["name"] for d in self.__dictionaries]}')

		self.__logger.info('Loading datasets')
		self.__datasets = self.__load_datasets()
		self.__logger.info(f'Loaded {len(self.__datasets)} datasets')

		self.__logger.info('Loading models')
		self.__models = self.__get_models()
		self.__logger.info(
			f'Loaded {len(self.__models)} '
			f'models: {[m.get_conf()["exp_name"] for m in self.__models]}')

		if 'results' in self.__conf:
			self.__logger.info(f'Loaded experiments has results')
			self.__results = [r.to_dict() for r in self.__conf.results]

		if 'labeled_dictionary_paths' in self.__conf:
			labeled_dict_paths = self.__conf.labeled_dictionary_paths
			self.__logger.info(f'Loading labeled dictionaries {len(labeled_dict_paths)}')
			self.__labeled_dictionary_dfs = [pd.read_csv(path, index_col=0) for path in labeled_dict_paths]
			self.__logger.info(f'Loaded labeled dictionaries')
		else:
			self.__labeled_dictionary_dfs = None
			self.__expanded_lexicons = None

	def run(self) -> bool:
		"""
		Run the end to end experiment for new ones or when overwrite_if_exists was chosen at initialization
		:return: boolean whether the experiment was run or not
		"""
		if self.__loaded_experiment:
			if not self.__overwrite_if_exists:
				self.__logger.info(
					'Skipping running the experiment as it is a loaded one. '
					'Use overwrite_if_exists to rerun.')
				return False
			else:
				self.__logger.warning(f'Overwriting experiment {self.__exp_name}')

		self.__lexicon = self.__get_lexicon()
		self.__logger.info(f'Using lexicon with configuration "{self.__lexicon.get_conf()}"')

		self.__dictionaries = self.__get_dictionaries()
		self.__logger.info(
			f'Using {len(self.__dictionaries)} dictionaries with names: '
			f'{[d.get_conf()["name"] for d in self.__dictionaries]}')

		self.__logger.info(f'Generating datasets for {len(self.__dictionaries)} dictionaries')
		self.__datasets = self.__get_datasets()

		self.__logger.info(f'Building {len(self.__dictionaries)} model(s)')
		self.__models = self.__get_models()

		self.__logger.info(f'Training and Evaluating models')
		self.__results = self.__get_results()
		self.__conf.results = self.__results

		if self.__do_label_dictionaries:
			self.label_dictionaries()
			self.expand_lexicons()
		else:
			self.__logger.info('Skipping labeling dictionaries')
			self.__logger.info('Skipping generating expanded lexicons')

		conf_path = self.__get_conf_path()
		self._save_conf(conf_path)
		self.__logger.info(f'Saved experiment configuration at {conf_path}')

		return True

	def get_results(self) -> List:
		return self.__results

	def label_dictionaries(self) -> List[pd.DataFrame]:
		if self.__labeled_dictionary_dfs is None:
			self.__logger.info('Labeling dictionaries')
			self.__labeled_dictionary_dfs = self.__get_output()

			self.__logger.info('Storing labeled dictionaries')
			output_paths = self.__store_output()
			self.__conf['labeled_dictionary_paths'] = output_paths
			self.__logger.info(f'Labeled dictionaries saved at {output_paths[0].parents}')

		return self.__labeled_dictionary_dfs

	def expand_lexicons(self) -> List[ExpandedLexicon]:
		if self.__expanded_lexicons is None:
			if self.__labeled_dictionary_dfs is None:
				raise ValueError('Labeled Dictionaries not initialized')

			self.__logger.info('Generating expanded lexicons')
			dict_names = [d.get_conf()['name'] for d in self.__dictionaries]

			def create_expanded_lexicon(name, df):
				df = df[df.prob_out >= self.__label_prob_threshold]
				df = df.set_index(Dictionary.WORD_COLUMN)
				term_to_label = df.label_out.to_dict()
				lex_name = f'{self.__exp_name}__{name}'
				lex = ExpandedLexicon(exp_name=lex_name,
									  term_to_label=term_to_label,
									  source_lexicon=self.__lexicon,
									  expanded_root_path=self.__expanded_root_path,
									  overwrite_if_exists=self.__overwrite_if_exists)
				save_path = lex.save()
				self.__logger.info(f'Saved lexicon for {name} at {save_path}')
			self.__expanded_lexicons = [create_expanded_lexicon(dict_name, labeled_df.copy())
										for dict_name, labeled_df in zip(dict_names, self.__labeled_dictionary_dfs)]

		return self.__expanded_lexicons

	def get_labeled_dictionaries(self):
		return self.__labeled_dictionary_dfs

	def get_expanded_lexicons(self):
		return self.__expanded_lexicons

	def __get_lexicon(self):
		name = None if 'name' not in self.__conf.lexicon else self.__conf.lexicon.name
		params = self.__conf.lexicon.to_dict().copy()
		params.pop('name', None)
		return get_lexicon(name=name, **params)

	def __get_dictionaries(self):
		def get_dictionary_from_conf(conf_dict):
			params = conf_dict.copy()
			name = params.pop('name')
			return get_dictionary(name=name, **params)
		return [get_dictionary_from_conf(conf.to_dict()) for conf in self.__conf.dictionaries]

	def __get_datasets(self):
		dataset_conf = self.__conf.dataset

		force_test_count = dataset_conf.force_test_count if 'force_test_count' in dataset_conf else True
		same_train_set = dataset_conf.same_train_set if 'same_train_set' in dataset_conf else False
		custom_root_path = Path(dataset_conf.custom_root_path) if 'custom_root_path' in dataset_conf else None
		quality_threshold = dataset_conf.quality_threshold if 'quality_threshold' in dataset_conf else None
		exclusions = dataset_conf.exclusions if 'exclusions' in dataset_conf else None

		generator = DatasetGenerator(exp_name=self.__exp_name,
									 lexicon=self.__lexicon,
									 dictionaries=self.__dictionaries,
									 test_count=dataset_conf.test_count,
									 force_test_count=force_test_count,
									 same_train_set=same_train_set,
									 quality_threshold=quality_threshold,
									 exclusions=exclusions,
									 dataset_root_path=custom_root_path)

		return generator.generate()

	def __load_datasets(self) -> List[Dataset]:
		datasets_conf = self.__conf.datasets
		return [Dataset(name=conf.name, datasets_root_path=conf.datasets_root_path) for conf in datasets_conf]

	def __get_models(self):
		name = self.__conf.model.name
		model_params = self.__conf.model.to_dict().copy()

		model_params.pop('name')
		if 'exp_name' not in model_params:
			model_params['exp_name'] = self.__exp_name
		if 'overwrite_if_exists' not in model_params:
			model_params['overwrite_if_exists'] = self.__overwrite_if_exists

		def get_model_for_dataset(dataset: Dataset):
			params = dict(model_params)
			params['dataset'] = dataset
			return get_model_by_name(name=name, **params)
		return [get_model_for_dataset(d) for d in self.__datasets]

	def __get_results(self):
		return [m.train_and_eval() for m in self.__models]

	def __get_output(self) -> List[pd.DataFrame]:
		return [model.apply(dictionary) for model, dictionary in zip(self.__models, self.__dictionaries)]

	def __get_default_labeled_output_path(self) -> Path:
		return Path(global_config.storage.root) / global_config.storage.labeled_out / self.__exp_name

	def __store_output(self) -> List[Path]:
		output_path = self._labeled_output_path \
			if self._labeled_output_path is not None else self.__get_default_labeled_output_path()
		output_path.mkdir(parents=True, exist_ok=self.__overwrite_if_exists)
		labeled_dictionary_paths = []
		for df, dictionary in zip(self.__labeled_dictionary_dfs, self.__dictionaries):
			file_path = output_path / f'{dictionary.get_conf()["name"]}.csv'
			df.to_csv(file_path)
			labeled_dictionary_paths.append(file_path)
		return labeled_dictionary_paths

	def __get_conf_path(self):
		exp_path = self.__experiments_root_paths / self.__exp_name
		exp_path.mkdir(parents=True, exist_ok=True)
		return exp_path / f'{self.__exp_name}__conf.yaml'

	def get_models(self) -> List[Model]:
		return self.__models

	def get_dictionaries(self) -> List[Dictionary]:
		return self.__dictionaries

	def get_conf(self) -> Dict[str, Any]:
		self.__conf.datasets = [d.get_conf() for d in self.__datasets]
		return self.__conf.to_primitives_dict()
