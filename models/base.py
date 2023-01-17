import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd

from common import ObjectWithConf, list_sorted_names_in_dir
from config import global_config, Config
from datasets.base import Dataset
from dictionaries import Dictionary


logging.basicConfig(level=global_config.logging)


def get_models_root_path() -> Path:
	return Path(global_config.storage.root) / global_config.storage.models_subdir


def list_model_names(models_root_path: Path = get_models_root_path()) -> List[str]:
	"""
	Returns a list of existing models
	:param models_root_path: Optional override for model root directory
	:return: Names as string list
	"""
	if models_root_path is None:
		models_root_path = get_models_root_path()
	return list_sorted_names_in_dir(models_root_path)


class ModelAlreadyExists(FileExistsError):
	pass


class ModelNotFoundError(FileNotFoundError):
	pass


class ModelInitializationError(ValueError):
	pass


class Model(ObjectWithConf, ABC):

	LABEL_OUT_COLUMN = 'label_out'
	PROB_OUT_COLUMN = 'prob_out'

	def __init__(self,
				 exp_name: str,
				 type_name: str,
				 dataset: Dataset = None,
				 overwrite_if_exists: bool = False,
				 models_root_path: Path = None,
				 datasets_root_path: Path = None):
		"""

		:param exp_name: Experiment name that helps identifies the saved instance belonging to which experiment
		:param type_name:  Technology or class this model is built using
		"""
		self._model_exists = exp_name in list_model_names(models_root_path)
		if not self._model_exists and dataset is None:
			raise ModelInitializationError(f'Either specify an existing model name '
										   f'or specify the dataset to train a new model')

		self._exp_name = exp_name
		self._models_root_path = models_root_path \
			if models_root_path is not None else get_models_root_path()
		self._datasets_root_path = datasets_root_path \
			if datasets_root_path is not None or dataset is None else dataset.get_conf()['datasets_root_path']
		self._overwrite_if_exists = overwrite_if_exists

		if self._model_exists:
			self._load()
		else:
			self._type_name = type_name
			self._dataset = dataset
			self._train_df = self._dataset.get_train()
			self._test_df = self._dataset.get_test()

			self._training_start_time = None
			self._creation_timestamp = None
			self._performance_result = None

		self._logger = logging.getLogger(__name__)

	def train_and_eval(self):
		if self._model_exists:
			if not self._overwrite_if_exists:
				raise ModelAlreadyExists(f'Model "{self._exp_name}" already exists in {self._models_root_path}')
			else:
				self._logger.warning(f'Going to overwrite existing "{self._exp_name}" model')

		start_time = datetime.now()
		self._logger.info(f'Training starting at {start_time}')
		training_happened = self._train()
		end_time = datetime.now()
		self._logger.info(f'Training ended at {end_time}')
		self._logger.info(f'Training duration {end_time - start_time}')

		if training_happened:
			self._training_start_time = start_time
			self._creation_timestamp = end_time
			self._performance_result = self._eval()
			self._save()

		return self._performance_result

	@abstractmethod
	def _train(self):
		pass

	@abstractmethod
	def _eval(self):
		pass

	@abstractmethod
	def _save(self):
		pass

	@abstractmethod
	def _load(self) -> Config:
		# Some parameters shouldn't be loaded and set only by initializer
		conf = Config(self._load_conf(self._get_conf_path()))

		self._type_name = conf.type_name
		self._models_root_path = Path(conf.models_root_path) \
			if conf.models_root_path is not None else get_models_root_path()
		self._datasets_root_path = Path(conf.datasets_root_path) \
			if conf.datasets_root_path is not None else None
		self._training_start_time = conf.training_start_time
		self._creation_timestamp = conf.creation_timestamp
		self._performance_result = conf.performance_result.to_dict()

		self._dataset = Dataset(conf.dataset.name, datasets_root_path=self._datasets_root_path)
		self._train_df = self._dataset.get_train()
		self._test_df = self._dataset.get_test()

		return conf

	@abstractmethod
	def get_test_labeled(self) -> pd.DataFrame:
		"""
		Returns the test set with model generated labels and probabilities
		:return: Dataframe
		"""
		pass

	@abstractmethod
	def apply(self, dictionary: Dictionary) -> pd.DataFrame:
		"""
		Apply a trained model on a dictionary
		:param dictionary:
		:return: Data frame with "label_out" and "prob_out" columns for label assigned by model and probability of
		assigned label
		"""
		pass

	def exists(self) -> bool:
		return self._model_exists

	def _get_conf_path(self):
		return self._models_root_path / self._exp_name / f'{self._exp_name}__conf.yaml'

	def get_conf(self):
		return {
			'exp_name':  self._exp_name,
			'type_name': self._type_name,
			'dataset': self._dataset.get_conf(),
			'overwrite_if_exists': self._overwrite_if_exists,
			'models_root_path': str(self._models_root_path),
			'datasets_root_path': str(self._datasets_root_path),
			'training_start_time': self._training_start_time,
			'creation_timestamp': self._creation_timestamp,
			'performance_result': self._performance_result
		}
