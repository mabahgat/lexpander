from typing import Optional

from models.base import Model
from models.text_transformers import BertClassifier


class InvalidModelName(ValueError):
	pass


def get_model_by_name(name: str, **kwargs) -> Optional[Model]:
	"""
	Gets a model instance by name. This only works for a specific predefined list of models
	:param name: Name of lexicon
	:param kwargs: parameters passed directly to the classifier initialization
	:return: An instance of lexicon if a corresponding type is found, else none
	"""
	if name == 'bert':
		return BertClassifier(**kwargs)
	else:
		raise InvalidModelName(f'Invalid model name "{name}"')
