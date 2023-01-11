import yaml

from experiments import Experiment
from utils import get_abs_file_path


def test_experiment_run_new_from_dict_by_name(tmp_path):
	root_path = tmp_path
	exp_root_path = root_path / 'experiments'
	models_path = root_path / 'models'
	exp_conf = {
		'exp_name': 'test_exp_by_name',
		'lexicon': {
			'name': 'liwc2015',
			'path': get_abs_file_path(__file__, 'resources/lexicons/test_liwc22_lookup.csv')
		},
		'dictionaries': [
			{
				'name': 'ud',
				'path': get_abs_file_path(__file__, 'resources/dictionaries/ud.csv')
			},
			{
				'name': 'wiktionary',
				'path': get_abs_file_path(__file__, 'resources/dictionaries/wiktionary.csv')
			}
		],
		'model': {
			'name': 'bert',
			'custom_root_path': models_path
		},
		'experiments_root_path': exp_root_path
	}
	exp = Experiment(exp_conf)
	exp.run()
	conf = exp.get_conf()

	assert conf['name'] == 'test_exp_by_name'
	assert conf['lexicon']['name'] == 'liwc2015'
	assert len(conf['dictionaries']) == 2
	assert conf['dictionaries'][0]['name'] == 'ud'
	assert conf['model']['name'] == 'bert'
	assert 'dataset' in conf
	assert 'result' in conf

	assert (exp_root_path / 'test_exp_by_name' / 'test_exp_by_name__conf.yaml').exists()


def test_experiment_run_new_from_dict_by_path(tmp_path):
	root_path = tmp_path
	exp_root_path = root_path / 'experiments'
	exp_root_path.mkdir()
	conf_path = exp_root_path / 'test_exp_by_path__conf.yaml'
	models_path = root_path / 'models'
	exp_conf = {
		'exp_name': 'test_exp_by_path',
		'lexicon': {
			'name': 'liwc2015',
			'path': str(get_abs_file_path(__file__, 'resources/lexicons/test_liwc22_lookup.csv'))
		},
		'dictionaries': [
			{
				'name': 'ud',
				'path': str(get_abs_file_path(__file__, 'resources/dictionaries/ud.csv'))
			},
			{
				'name': 'wiktionary',
				'path': str(get_abs_file_path(__file__, 'resources/dictionaries/wiktionary.csv'))
			}
		],
		'model': {
			'name': 'bert',
			'custom_root_path': str(models_path)
		},
		'experiments_root_path': str(exp_root_path)
	}

	with open(conf_path, mode='w') as conf_file:
		yaml.dump(exp_conf, conf_file)

	exp = Experiment(conf_path=conf_path)
	conf = exp.get_conf()

	assert conf['exp_name'] == 'test_exp_by_path'
	assert conf['lexicon']['name'] == 'liwc2015'
	assert len(conf['dictionaries']) == 2
	assert conf['dictionaries'][0]['name'] == 'ud'
	assert conf['model']['name'] == 'bert'
