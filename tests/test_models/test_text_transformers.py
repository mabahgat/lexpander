from datasets.base import Dataset
from dictionaries import SimpleDictionary
from models.base import list_model_names
from models.text_transformers import BertClassifier
from tests.utils import get_abs_file_path


def __get_dataset(datasets_root_path):
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

	dataset_name = 'rand_3_labels_150_examples'
	return Dataset(name=dataset_name,
				   datasets_root_path=datasets_root_path)


def test_bert_classifier(tmp_path):
	# Multiple checks in the same test case for large setup time
	datasets_root_path = get_abs_file_path(__file__, '../resources/datasets/')
	dataset = __get_dataset(datasets_root_path)

	models_root_path = tmp_path
	model = BertClassifier(exp_name='model_test',
						   dataset=dataset,
						   epochs=1,
						   models_root_path=models_root_path)

	results = model.train_and_eval()

	assert 'f_score' in results
	assert type(results['f_score']) is float
	assert 'accuracy' in results
	assert type(results['accuracy']) is float
	assert 'report' in results
	assert type(results['report']) is dict

	test_df = model.get_test_labeled()

	assert len(test_df) == len(dataset.get_test())
	assert 'label_out' in test_df
	assert 'prob_out' in test_df

	conf = model.get_conf()

	assert conf['epochs'] == 1
	assert conf['learning_rate'] == 5e-5
	assert set(conf['label_2_idx'].keys()) == {'label_1', 'label_2', 'label_3'}

	assert conf['exp_name'] == 'model_test'
	assert conf['type_name'] == 'TFBertForSequenceClassification__bert-base-uncased'

	assert conf['dataset'] == dataset.get_conf()
	assert conf['models_root_path'] == str(models_root_path)
	assert conf['datasets_root_path'] == str(datasets_root_path)
	assert 'training_start_time' in conf
	assert 'creation_timestamp' in conf
	assert 'performance_result' in conf
	assert 'f_score' in conf['performance_result']
	assert 'accuracy' in conf['performance_result']
	assert 'report' in conf['performance_result']

	assert (models_root_path / 'model_test').exists()
	assert (models_root_path / 'model_test' / 'tf_model.h5').exists()
	assert (models_root_path / 'model_test' / 'config.json').exists()
	assert (models_root_path / 'model_test' / 'model_test__conf.yaml').exists()
	assert (models_root_path / 'model_test' / 'model_test__test_out.csv').exists()

	ud_path = get_abs_file_path(__file__, '../resources/dictionaries/ud.csv')
	dictionary = SimpleDictionary('ud', ud_path)
	labeled_df = model.apply(dictionary)

	assert 'label_out' in labeled_df
	assert 'prob_out' in labeled_df

	assert 'model_test' in list_model_names(models_root_path)

	loaded_model = BertClassifier(exp_name='model_test',
								  models_root_path=models_root_path,
								  dataset_root_path=datasets_root_path)
	loaded_conf = loaded_model.get_conf()
	loaded_result = conf['performance_result']

	assert loaded_conf['epochs'] == 1
	assert loaded_conf['learning_rate'] == 5e-5
	assert set(loaded_conf['label_2_idx'].keys()) == {'label_1', 'label_2', 'label_3'}

	assert loaded_conf['exp_name'] == 'model_test'
	assert loaded_conf['type_name'] == 'TFBertForSequenceClassification__bert-base-uncased'
	assert loaded_conf['datasets_root_path'] == str(datasets_root_path)

	assert 'f_score' in loaded_result
	assert type(loaded_result['f_score']) is float
	assert 'accuracy' in loaded_result
	assert type(loaded_result['accuracy']) is float
	assert 'report' in loaded_result
	assert type(loaded_result['report']) is dict

	assert loaded_conf['dataset'] == dataset.get_conf()
	assert loaded_conf['models_root_path'] == str(models_root_path)
	assert 'training_start_time' in loaded_conf
	assert loaded_conf['training_start_time'] == conf['training_start_time']
	assert 'creation_timestamp' in loaded_conf
	assert loaded_conf['creation_timestamp'] == conf['creation_timestamp']
	assert 'performance_result' in loaded_conf
	assert loaded_conf['performance_result'] == conf['performance_result']
