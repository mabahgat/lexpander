import yaml

from experiments import Experiment
from utils import get_abs_file_path


def test_experiment_run(tmp_path):
	root_path = tmp_path
	exp_root_path = root_path / 'experiments'
	models_path = root_path / 'models'
	datasets_path = root_path / 'datasets'
	labeled_output_path = root_path / 'out'
	expanded_root_path = root_path / 'expanded'
	label_prob_threshold = 0

	lexicon_path = get_abs_file_path(__file__, 'resources/lexicons/rand_3_labels_150_examples.csv')

	exp_conf = {
		'exp_name': 'test_exp_by_name',
		'lexicon': {
			'csv_path': lexicon_path
		},
		'dictionaries': [
			{
				'name': 'rand_3_labels_150_examples_1',
				'file_path': get_abs_file_path(__file__, 'resources/dictionaries/rand_3_labels_150_examples.csv')
			},
			{
				'name': 'rand_3_labels_150_examples_2',
				'file_path': get_abs_file_path(__file__, 'resources/dictionaries/rand_3_labels_150_examples.csv')
			}
		],
		'model': {
			'name': 'bert',
			'models_root_path': models_path,
			'epochs': 1
		},
		'dataset': {
			'test_count': 10,
			'force_test_count': True,
			'custom_root_path': datasets_path,
			'exclusions': []
		},
		'experiments_root_path': exp_root_path,
		'labeled_output_path': labeled_output_path,
		'label_prob_threshold': label_prob_threshold,
		'expanded_root_path': expanded_root_path,
		'do_label_dictionaries': True
	}
	exp = Experiment(exp_conf)
	exp.run()
	conf = exp.get_conf()

	assert conf['exp_name'] == 'test_exp_by_name'
	assert conf['lexicon']['csv_path'] == str(lexicon_path)
	assert len(conf['dictionaries']) == 2
	assert conf['dictionaries'][0]['name'] == 'rand_3_labels_150_examples_1'
	assert conf['model']['name'] == 'bert'
	assert 'dataset' in conf
	assert 'results' in conf

	assert (exp_root_path / 'test_exp_by_name' / 'test_exp_by_name__conf.yaml').exists()
	assert len(list(labeled_output_path.glob('*'))) == 2
	assert len(list(expanded_root_path.glob('*'))) == 2

	# Testing loading here given the setup time it takes

	loaded_conf = {
		'exp_name': 'test_exp_by_name',
		'experiments_root_path': exp_root_path
	}
	loaded_exp = Experiment(loaded_conf)
	did_run = loaded_exp.run()

	assert did_run == False
	assert len(loaded_exp.get_models()) == 2
	assert len(loaded_exp.label_dictionaries()) == 2


def test_experiment_run_with_overwrite(tmp_path):
	root_path = tmp_path
	exp_root_path = root_path / 'experiments'
	models_path = root_path / 'models'
	datasets_path = root_path / 'datasets'
	labeled_output_path = root_path / 'out'
	expanded_root_path = root_path / 'expanded'
	label_prob_threshold = 0

	lexicon_path = get_abs_file_path(__file__, 'resources/lexicons/rand_3_labels_150_examples.csv')

	exp_conf = {
		'exp_name': 'test_exp_by_name',
		'lexicon': {
			'csv_path': lexicon_path
		},
		'dictionaries': [
			{
				'name': 'rand_3_labels_150_examples_1',
				'file_path': get_abs_file_path(__file__, 'resources/dictionaries/rand_3_labels_150_examples.csv')
			},
			{
				'name': 'rand_3_labels_150_examples_2',
				'file_path': get_abs_file_path(__file__, 'resources/dictionaries/rand_3_labels_150_examples.csv')
			}
		],
		'model': {
			'name': 'bert',
			'models_root_path': models_path,
			'epochs': 1
		},
		'dataset': {
			'test_count': 10,
			'force_test_count': True,
			'custom_root_path': datasets_path,
			'exclusions': []
		},
		'experiments_root_path': exp_root_path,
		'labeled_output_path': labeled_output_path,
		'label_prob_threshold': label_prob_threshold,
		'expanded_root_path': expanded_root_path,
		'do_label_dictionaries': True,
		'overwrite_if_exists': False
	}
	exp = Experiment(exp_conf)
	exp.run()

	loaded_conf = {
		'exp_name': 'test_exp_by_name',
		'experiments_root_path': exp_root_path,
		'overwrite_if_exists': True
	}
	loaded_exp = Experiment(loaded_conf)
	did_run = loaded_exp.run()

	assert did_run == True
	assert len(loaded_exp.get_models()) == 2
	assert len(loaded_exp.label_dictionaries()) == 2
	assert len(list(labeled_output_path.glob('*'))) == 2
	assert len(list(expanded_root_path.glob('*'))) == 2


def test_experiment_run_new_from_dict_by_path(tmp_path):
	root_path = tmp_path
	exp_root_path = root_path / 'experiments'
	exp_root_path.mkdir()
	conf_path = exp_root_path / 'test_exp_by_path__conf.yaml'
	models_path = root_path / 'models'
	datasets_path = root_path / 'datasets'
	labeled_output_path = root_path / 'out'

	lexicon_path = get_abs_file_path(__file__, 'resources/lexicons/rand_3_labels_150_examples.csv')

	exp_conf = {
		'exp_name': 'test_exp_by_path',
		'lexicon': {
			'csv_path': str(lexicon_path)
		},
		'dictionaries': [
			{
				'name': 'rand_3_labels_150_examples_1',
				'path': str(get_abs_file_path(__file__, 'resources/dictionaries/rand_3_labels_150_examples.csv'))
			},
			{
				'name': 'rand_3_labels_150_examples_2',
				'path': str(get_abs_file_path(__file__, 'resources/dictionaries/rand_3_labels_150_examples.csv'))
			}
		],
		'model': {
			'name': 'bert',
			'models_root_path': str(models_path),
			'epochs': 1
		},
		'dataset': {
			'test_count': 10,
			'force_test_count': True,
			'custom_root_path': str(datasets_path),
			'exclusions': []
		},
		'experiments_root_path': str(exp_root_path),
		'labeled_output_path': str(labeled_output_path)
	}

	with open(conf_path, mode='w') as conf_file:
		yaml.dump(exp_conf, conf_file)

	exp = Experiment(conf_path=conf_path)
	conf = exp.get_conf()

	assert conf['exp_name'] == 'test_exp_by_path'
	assert len(conf['dictionaries']) == 2
	assert conf['dictionaries'][0]['name'] == 'rand_3_labels_150_examples_1'
	assert conf['model']['name'] == 'bert'
