import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Optional, List

import pandas as pd

from common import ObjectWithConf, list_sorted_names_in_dir
from config import global_config, Config
from datasets.base import Dataset
from dictionaries import Dictionary


logging.basicConfig(level=global_config.logging)


def get_models_root_path() -> Path:
	return Path(global_config.storage.root) / global_config.storage.datasets_subdir


def list_model_names(models_root_path: Path = get_models_root_path()) -> List[str]:
	"""
	Returns a list of existing models
	:param models_root_path: Optional override for model root directory
	:return: Names as string list
	"""
	return list_sorted_names_in_dir(models_root_path)


class ModelAlreadyExists(FileExistsError):
	pass


class ModelNotFoundError(FileNotFoundError):
	pass


class ModelInitializationError(ValueError):
	pass


class Model(ObjectWithConf, ABC):

	def __init__(self,
				 exp_name: str,
				 type_name: str,
				 dataset: Optional[Dataset],
				 overwrite_if_exists: bool = False,
				 models_root_path: Path = None):
		"""

		:param exp_name: Experiment name that helps identifies the saved instance belonging to which experiment
		:param type_name:  Technology or class this model is built using
		"""
		self._exp_name = exp_name
		self._models_root_path = models_root_path if models_root_path is not None else get_models_root_path()

		if self._model_already_exists():
			self._load()
		else:
			self._type_name = type_name
			self._dataset = dataset
			self._train_df = self._dataset.get_train()
			self._test_df = self._dataset.get_test()
			self._overwrite_if_exists = overwrite_if_exists

			self._training_start_time = None
			self._creation_timestamp = None
			self._performance_result = None

		self._logger = logging.getLogger(__name__)

	def train_and_eval(self):
		if self._model_already_exists():
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

			self._save()

		self._performance_result = self._eval()
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
		conf = Config(self._load_conf(self._get_conf_path()))
		self._type_name = conf.type_name
		self._dataset = Dataset(conf.dataset.name)
		self._train_df = self._dataset.get_train()
		self._test_df = self._dataset.get_test()
		self._overwrite_if_exists = conf.overwrite_if_exists
		self._model_root_path = conf.model_root_path
		self._training_start_time = conf.training_start_time
		self._overwrite_if_exists = conf.overwrite_if_exists
		self._performance_result = conf.performance_result
		return conf

	@abstractmethod
	def apply(self, dictionary: Dictionary, skip_labeled: bool = True) -> pd.DataFrame:
		"""
		Apply a trained model on a dictionary
		:param dictionary:
		:param skip_labeled: If there are labeled entries skip them
		:return: Data frame with "label_out" and "prob_out" columns for label assigned by model and probability of
		assigned label
		"""
		pass

	def _get_conf_path(self):
		return self._models_root_path / self._exp_name / f'{self._exp_name}__conf.yaml'

	def _model_already_exists(self):
		return self._exp_name in list_model_names(self._models_root_path)

	def get_conf(self):
		return {
			'exp_name':  self._exp_name,
			'type_name': self._type_name,
			'dataset': self._dataset.get_conf(),
			'overwrite_if_exists': self._overwrite_if_exists,
			'models_root_path': self._models_root_path,
			'training_start_time': self._training_start_time,
			'creation_timestamp': self._creation_timestamp,
			'performance_result': self._performance_result
		}
