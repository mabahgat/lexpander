from pathlib import Path

import pandas as pd
import pytest

from datasets.base import Dataset
from models.transformers import BertClassifier
from models.utils import get_model_by_name, InvalidModelName


def get_sample_dataset(datasets_root_path: Path):
	data = {'words': ['me', 'you'], 'label': ['self', 'other']}
	train_df = pd.DataFrame.from_dict(data)
	valid_df = pd.DataFrame.from_dict(data)
	test_df = pd.DataFrame.from_dict(data)

	return Dataset(name='test_dataset',
				   train_df=train_df,
				   valid_df=valid_df,
				   test_df=test_df,
				   source_path='/some/path',
				   datasets_root_path=datasets_root_path)


def test_get_model_by_name_or_none(tmp_path):
	dataset_root_path = tmp_path / 'datasets'
	dataset_root_path.mkdir()
	dataset = get_sample_dataset(dataset_root_path)

	bert = get_model_by_name(name='bert', exp_name='test_exp', dataset=dataset)
	assert type(bert) == BertClassifier

	with pytest.raises(InvalidModelName):
		get_model_by_name(name='blah', exp_name='test_exp')
