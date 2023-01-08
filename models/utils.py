from pathlib import Path
from typing import Optional

from datasets.base import Dataset
from models.base import Model, get_models_root_path
from models.transformers import BertClassifier


class InvalidModelName(ValueError):
	pass


def get_model_by_name(name: str,
					  exp_name: str,
					  dataset: Dataset = None,
					  models_root_path: Path = get_models_root_path()) -> Optional[Model]:
	"""
	Gets a model instance by name. This only works for a specific predefined list of models
	:param name: Name of lexicon
	:param exp_name: experiment name string
	:param dataset: dataset used to train and eval a model. Only required if model belongs to a new experiment
	:param models_root_path: custom models root_path
	:return: An instance of lexicon if a corresponding type is found, else none
	"""
	if name == 'bert':
		return BertClassifier(exp_name=exp_name, dataset=dataset, models_root_path=models_root_path)
	else:
		raise InvalidModelName(f'Invalid model name "{name}"')
