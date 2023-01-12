import logging
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd

from common import ObjectWithConf, list_sorted_names_in_dir
from config import global_config, Config
from datasets.base import Dataset
from datasets.generators import DatasetGenerator
from dictionaries import get_dictionary
from lexicons import get_lexicon
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

		self.__config_dict = ObjectWithConf._load_conf(conf_path) if conf_path is not None else conf
		self.__conf = Config(self.__config_dict)

		self.__exp_name = self.__conf.exp_name
		self.__experiments_root_paths = Path(self.__conf.experiments_root_path) \
			if 'experiments_root_path' in self.__conf else get_experiments_root_path()

		self._expanded_output_path = Path(self.__conf.expanded_output_path) \
			if 'expanded_output_path' in self.__conf else None

		if self.__exists():
			self.__load()
		else:
			self.__lexicon = None
			self.__dictionaries = None
			self.__datasets = None
			self.__models = None
			self.__results = None
			self.__labeled_dictionary_dfs = None

		self.__logger = logging.getLogger(__name__)

	def __exists(self):
		return self.__exp_name in list_experiment_names(self.__experiments_root_paths)

	def __load(self):
		raise NotImplementedError()

	def run(self):
		self.__lexicon = self.__get_lexicon()
		self.__logger.info(f'Using lexicon with configuration "{self.__lexicon.get_conf()}"')

		self.__dictionaries = self.__get_dictionaries()
		self.__logger.info(f'Using {len(self.__dictionaries)} dictionaries with names: '
						   f'{[d.get_conf()["name"] for d in self.__dictionaries]}')

		self.__logger.info(f'Generating datasets for {len(self.__dictionaries)} dictionaries')
		self.__datasets = self.__get_datasets()

		self.__logger.info(f'Building {len(self.__dictionaries)} model(s)')
		self.__models = self.__get_models()

		self.__logger.info(f'Training and Evaluating models')
		self.__results = self.__get_results()
		self.__config_dict['results'] = self.__results

		self.__logger.info('Labeling dictionaries')
		self.__labeled_dictionary_dfs = self.__get_output()

		self.__logger.info('Storing labeled dictionaries')
		output_path = self.__store_output()
		self.__logger.info(f'Labeled dictionaries saved at {output_path}')

		conf_path = self.__get_conf_path()
		self._save_conf(conf_path)
		self.__logger.info(f'Saved experiment configuration at {conf_path}')

	def get_results(self):
		return self.__results

	def get_labeled_dictionaries(self):
		return self.__labeled_dictionary_dfs

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
		return [get_dictionary_from_conf(conf_dict) for conf_dict in self.__conf.dictionaries]  # because .dictionaries is an array objects are dict not Config

	def __get_datasets(self):
		dataset_conf = self.__conf.dataset

		force_test_count = dataset_conf.force_test_count if 'force_test_count' in dataset_conf else True
		same_train_set = dataset_conf.same_train_set if 'same_train_set' in dataset_conf else False
		custom_root_path = Path(dataset_conf.custom_root_path) if 'custom_root_path' in dataset_conf else None
		exclusions = dataset_conf.exclusions if 'exclusions' in dataset_conf else None

		generator = DatasetGenerator(exp_name=self.__exp_name,
									 lexicon=self.__lexicon,
									 dictionaries=self.__dictionaries,
									 test_count=dataset_conf.test_count,
									 force_test_count=force_test_count,
									 same_train_set=same_train_set,
									 exclusions=exclusions,
									 dataset_root_path=custom_root_path)

		return generator.generate()

	def __get_models(self):
		name = self.__conf.model.name
		model_params = self.__conf.model.to_dict().copy()
		if 'exp_name' not in model_params:
			model_params['exp_name'] = self.__exp_name
		model_params.pop('name')

		def get_model_for_dataset(dataset: Dataset):
			params = dict(model_params)
			params['dataset'] = dataset
			return get_model_by_name(name=name, **params)
		return [get_model_for_dataset(d) for d in self.__datasets]

	def __get_results(self):
		return [m.train_and_eval() for m in self.__models]

	def __get_output(self) -> List[pd.DataFrame]:
		return [model.apply(dictionary) for model, dictionary in zip(self.__models, self.__dictionaries)]

	def __get_default_expanded_output_path(self) -> Path:
		return Path(global_config.storage.root) / global_config.storage.expand_out / self.__exp_name

	def __store_output(self) -> Path:
		output_path = self._expanded_output_path \
			if self._expanded_output_path is not None else self.__get_default_expanded_output_path()
		output_path.mkdir(parents=True)
		for df, dictionary in zip(self.__labeled_dictionary_dfs, self.__dictionaries):
			file_path = output_path / f'{dictionary.get_conf()["name"]}.csv'
			df.to_csv(file_path)
		return output_path

	def __get_conf_path(self):
		exp_path = self.__experiments_root_paths / self.__exp_name
		exp_path.mkdir(parents=True)
		return exp_path / f'{self.__exp_name}__conf.yaml'

	def get_conf(self) -> Dict[str, Any]:
		return self.__config_dict
