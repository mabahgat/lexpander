from pathlib import Path
from typing import List

import pandas as pd
import numpy as np
from transformers import TFBertForSequenceClassification, TFBertTokenizer
import tensorflow as tf
from sklearn import metrics

from datasets.base import Dataset
from dictionaries import Dictionary
from models.base import Model


class BertClassifier(Model):
	DEFAULT_PRETRAINED_MODEL_NAME = 'bert-base-uncased'
	DEFAULT_LEARNING_RATE = 5e-5
	DEFAULT_EPOCHS = 5

	def __init__(self,
				 exp_name: str,
				 dataset: Dataset = None,
				 pretrained_model_name: str = DEFAULT_PRETRAINED_MODEL_NAME,
				 learning_rate: float = DEFAULT_LEARNING_RATE,
				 epochs: int = DEFAULT_EPOCHS,
				 overwrite_if_exists: bool = False,
				 models_root_path: Path = None,
				 dataset_root_path: Path = None):
		"""
		Creates a new empty model that should be trained
		:param exp_name: Names of the experiment being run. Should be the same as the value passed to DatasetGenerator
		:param dataset:
		:param pretrained_model_name:
		:param learning_rate:
		:param epochs:
		:param models_root_path: Optional override for where models are stored
		"""
		model_type = f'{TFBertForSequenceClassification.__name__}__{pretrained_model_name}'
		models_root_path = Path(models_root_path) if models_root_path is not None else None
		dataset_root_path = Path(dataset_root_path) if dataset_root_path is not None else None
		super().__init__(exp_name=exp_name,
						 type_name=model_type,
						 dataset=dataset,
						 overwrite_if_exists=overwrite_if_exists,
						 models_root_path=models_root_path,
						 datasets_root_path=dataset_root_path)

		if not self._model_exists:
			self._pretrained_model_name = pretrained_model_name
			self._learning_rate = learning_rate
			self._epochs = epochs

			self._label_2_idx, self._idx_2_label = self.__get_labels()
			self._tokenizer = TFBertTokenizer.from_pretrained(pretrained_model_name)
			self._model = None

			self.__cached_performance_result = None
			self.__cached_test_labels = None
			self.__cached_test_label_probs = None

	def __get_labels(self):
		train_labels = set(self._train_df.label.to_list())
		test_labels = set(self._test_df.label.to_list())

		if len(set(train_labels) & set(test_labels)) != len(train_labels):
			self._logger.warning(f'Labels in the training and testing are different.'
								  f'Training labels {train_labels}.'
								  f'Test labels {test_labels}')

		labels = set(set(train_labels) | set(test_labels))
		label_2_idx = {l: i for i, l in enumerate(labels)}
		idx_2_label = {i: l for i, l in enumerate(labels)}

		self._logger.info(f'Number of labels in the dataset {len(labels)}. Labels are: {labels}')

		return label_2_idx, idx_2_label

	def __encode_set(self, df: pd.DataFrame):
		x = self.__encode_text(df.text.to_list())

		label_lst = df.label.to_list()
		y = np.array([self._label_2_idx[label] for label in label_lst])
		return x, y

	def __encode_text(self, text: List[str]):
		return self._tokenizer(text, truncation=True, padding='max_length')

	def _train(self) -> bool:
		"""
		Train model
		:return: whether model was trained (weights updated or not)
		"""
		if self._model is not None:
			self._logger.info('Model already trained, skipping training step')
			return False
		else:
			train_x, train_y = self.__encode_set(self._train_df)

			model = self.__build_model()
			model.fit(train_x, train_y, epochs=self._epochs)
			self._model = model
			return True

	def __build_model(self):
		model = TFBertForSequenceClassification.from_pretrained(self._pretrained_model_name,
																num_labels=len(self._label_2_idx.keys()))
		optimizer = tf.keras.optimizers.Adam(learning_rate=self._learning_rate)
		model.compile(optimizer=optimizer)
		return model

	def _eval(self):
		if self.__cached_performance_result is None:
			test_x, test_y = self.__encode_set(self._test_df)
			test_y_label_out, test_y_out_prob, test_y_out = self.__apply(test_x)

			self.__cached_test_labels = test_y_label_out
			self.__cached_test_label_probs = test_y_out_prob

			f_score = float(metrics.f1_score(test_y, test_y_out, average='macro'))
			accuracy = float(metrics.accuracy_score(test_y, test_y_out))
			report = metrics.classification_report(test_y,
												   test_y_out,
												   labels=list(self._idx_2_label.keys()),
												   target_names=list(self._idx_2_label.values()),
												   output_dict=True)
			self.__cached_performance_result = {
				'f_score': f_score,
				'accuracy': accuracy,
				'report': report
			}

		return self.__cached_performance_result

	def __apply(self, encoded_text) -> (List[str], List[float], List[int]):
		"""
		Applies a model to encoded text
		:param encoded_text: encoded (tokenized in the case of BERT) ready for the model to consume
		:return: list of labels, list of probabilities, list of labels but in the form of indexes
		"""
		probs = tf.nn.softmax(self._model.predict(encoded_text).logits).numpy().tolist()
		label_indexes = [np.argmax(o) for o in probs]
		labels = [self._idx_2_label[index] for index in label_indexes]
		return labels, probs, label_indexes

	def __model_path(self) -> Path:
		return self._models_root_path / self._exp_name

	def _save(self):
		model_path = self.__model_path()
		self._model.save_pretrained(model_path)
		self._save_conf(self._get_conf_path())
		self.__save_test_labels()
		self._logger.info(f'Model saved to {model_path}')

	def __get_test_results_path(self):
		"""
		Path to store test labels and probabilities assigned by the model
		:return:
		"""
		return self.__model_path() / f'{self._exp_name}__test_out.csv'

	def __save_test_labels(self):
		test_df = self.get_test_labeled()
		test_df.to_csv(self.__get_test_results_path())

	def _load(self):
		conf = super()._load()
		self._epochs = conf.epochs
		self._learning_rate = conf.learning_rate
		self._pretrained_model_name = conf.pretrained_model_name

		self._label_2_idx = conf.label_2_idx.to_dict()
		self._idx_2_label = {index: label for label, index in self._label_2_idx.items()}

		self._tokenizer = TFBertTokenizer.from_pretrained(self._pretrained_model_name)
		self._model = self.__load_model()

		self.__load_test_labels()

	def __load_test_labels(self):
		test_df = pd.read_csv(self.__get_test_results_path(), index_col=0)
		self.__cached_test_labels = test_df.label_out.to_list()
		self.__cached_test_label_probs = test_df.prob_out.to_list()

	def __load_model(self):
		model = self.__build_model()
		weights_file_path = self.__model_path() / 'tf_model.h5'
		model.load_weights(weights_file_path)
		return model

	def get_test_labeled(self) -> pd.DataFrame:
		if self.__cached_test_labels is None or self.__cached_test_label_probs is None:
			self._eval()
		test_df = self._test_df.copy(deep=True)
		test_df[Model.LABEL_OUT_COLUMN] = self.__cached_test_labels
		test_df[Model.PROB_OUT_COLUMN] = self.__cached_test_label_probs
		return test_df

	def apply(self, dictionary: Dictionary) -> pd.DataFrame:
		df = dictionary.get_all_records()
		text = df.text.to_list()
		encoded_text = self.__encode_text(text)
		labels, probs, _ = self.__apply(encoded_text)
		df[Model.LABEL_OUT_COLUMN] = labels
		df[Model.PROB_OUT_COLUMN] = probs
		return df

	def get_conf(self):
		conf = super().get_conf()
		conf['epochs'] = self._epochs
		conf['learning_rate'] = self._learning_rate
		conf['pretrained_model_name'] = self._pretrained_model_name
		conf['label_2_idx'] = self._label_2_idx
		return conf
