from datasets.base import Dataset
from models.transformers import BertClassifier
from tests.utils import get_abs_file_path


def __get_dataset():
	"""
	Label Distribution for this file
	label_1	65
	label_2	63
	label_3	53
	label_4	51
	label_5	68
	Average quality = 3.68667
	:return:
	"""
	dataset_root_path = get_abs_file_path(__file__, '../resources/datasets/')
	dataset_name = 'rand_3_labels_150_examples'
	return Dataset(name=dataset_name,
				   datasets_root_path=dataset_root_path)


def test_bert_classifier_conf(tmp_path):
	dataset = __get_dataset()
	models_root_path = tmp_path
	model = BertClassifier(exp_name='model_test',
						   dataset=dataset,
						   epochs=1,
						   models_root_path=models_root_path)

	results = model.train_and_eval()

	assert 'f_score' in results
	assert 'accuracy' in results
	assert 'report' in results

	conf = model.get_conf()

	assert conf['epochs'] == 1
	assert conf['learning_rate'] == 5e-5
	assert set(conf['label_2_idx'].keys()) == {'label_1', 'label_2', 'label_3'}

	assert conf['exp_name'] == 'model_test'
	assert conf['type_name'] == 'TFBertForSequenceClassification__bert-base-uncased'

	assert conf['dataset'] == dataset.get_conf()
	assert conf['models_root_path'] == models_root_path
	assert 'training_start_time' in conf
	assert 'creation_timestamp' in conf
	assert 'performance_result' in conf

	assert (models_root_path / 'model_test').exists()
	assert (models_root_path / 'model_test' / 'tf_model.h5').exists()
	assert (models_root_path / 'model_test' / 'config.json').exists()
	assert (models_root_path / 'model_test' / 'model_test__conf.yaml').exists()
