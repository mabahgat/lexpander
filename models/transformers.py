from pathlib import Path

import pandas as pd
import numpy as np
from transformers import TFBertForSequenceClassification, TFBertTokenizer
import tensorflow as tf
from sklearn import metrics

from datasets.base import Dataset
from dictionaries import Dictionary
from models.base import Model, list_model_names, ModelInitializationError


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
				 models_root_path: Path = None):
		"""
		Creates a new empty model that should be trained
		:param exp_name: Names of the experiment being run. Should be the same as the value passed to DatasetGenerator
		:param dataset:
		:param pretrained_model_name:
		:param learning_rate:
		:param epochs:
		:param models_root_path: Optional override for where models are stored
		"""
		if exp_name not in list_model_names() and dataset is None:
			raise ModelInitializationError(f'Either specify an existing model name '
										   f'or specify the dataset to train a new model')

		model_type = f'{TFBertForSequenceClassification.__name__}__{pretrained_model_name}'
		super().__init__(exp_name=exp_name,
						 type_name=model_type,
						 dataset=dataset,
						 overwrite_if_exists=overwrite_if_exists,
						 models_root_path=models_root_path)

		if not self._model_already_exists():
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
		train_labels = self._train_df.label.to_list()
		test_labels = self._test_df.label.to_list()

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
		text_lst = df.text.to_list()
		x = self._tokenizer.encode_plus(text_lst, truncation=True, padding='max_length')

		label_lst = df.label.to_list()
		y = np.array([self._label_2_idx[label] for label in label_lst])
		return x, y

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
			test_y_out_prob = tf.nn.softmax(self._model.predict(test_x).logits).numpy().tolist()
			test_y_out = [np.argmax(o) for o in test_y_out_prob]
			test_y_label_out = [self._idx_2_label[index] for index in test_y_out]

			self.__cached_test_labels = test_y_label_out
			self.__cached_test_label_probs = test_y_out_prob

			f_score = metrics.f1_score(test_y, test_y_out, average='macro')
			accuracy = metrics.accuracy_score(test_y, test_y_out)
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

	def __model_path(self) -> Path:
		return self._models_root_path / self._exp_name

	def _save(self):
		model_path = self.__model_path()
		self._model.save_pretrained(model_path)
		self._save_conf(self._get_conf_path())
		self._logger.info(f'Model saved to {model_path}')

	def _load(self):
		conf = super()._load_conf(self._get_conf_path())
		self._epochs = conf.epochs
		self._learning_rate = conf.learing_rate
		self._pretrained_model_name = conf.pretrained_model_name

		self._label_2_idx = conf.label_2_idx
		self._idx_2_label = {index: label for label, index in self._label_2_idx.items()}

		self._tokenizer = TFBertTokenizer.from_pretrained(self._pretrained_model_name)
		self._model = self.__load_model()

	def __load_model(self):
		model = self.__build_model()
		weights_file_path = self.__model_path() / 'tf_model.h5'
		model.load_weights(weights_file_path)
		return model

	def apply(self, dictionary: Dictionary, skip_labeled: bool = True) -> pd.DataFrame:
		pass

	def get_conf(self):
		conf = super().get_conf()
		conf['epochs'] = self._epochs
		conf['learning_rate'] = self._learning_rate
		conf['pretrained_model_name'] = self._pretrained_model_name
		conf['label_2_idx'] = self._label_2_idx
		return conf
