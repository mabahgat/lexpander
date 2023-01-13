from config import Config, load_config
from tests.utils import get_abs_file_path


def test_config():
    dict_data = {
        'param1': 'value1',
        'param2': 'value2',
        'param3': {
            'param3_1': 'value3.1',
            'param3_2': {
                'param3_2_1': 'value3.2.1'
            }
        }
    }
    config = Config(config_dict=dict_data)

    assert config['param1'] == 'value1'
    assert config.param1 == 'value1'
    assert config['param3']['param3_1'] == 'value3.1'
    assert config.param3.param3_1 == 'value3.1'
    assert config['param3']['param3_2']['param3_2_1'] == 'value3.2.1'
    assert config.param3.param3_2.param3_2_1 == 'value3.2.1'


def test_load_config():
    config_path = get_abs_file_path(__file__, 'resources/test_config.yaml')
    config = load_config(config_path)

    assert config.version == 0.1
    assert config.paths.lex == '/path/to/lex'
    assert config.values.lex.secondary == 'two'


def test_contains():
    dict_data = {
        'param1': 'value1',
        'param2': 'value2',
        'param3': {
            'param3_1': 'value3.1',
            'param3_2': {
                'param3_2_1': 'value3.2.1'
            }
        }
    }
    config = Config(config_dict=dict_data)

    assert 'param1' in config
    assert 'param4' not in config
    assert 'param3_1' in config.param3
    assert 'param3_3' not in config.param3


def test_to_dict():
    dict_data = {
        'param1': 'value1',
        'param2': 'value2',
        'param3': {
            'param3_1': 'value3.1',
            'param3_2': {
                'param3_2_1': 'value3.2.1'
            }
        }
    }
    config = Config(config_dict=dict_data)

    assert type(config) is Config
    assert type(config.to_dict()) is dict
    assert type(config.param3.to_dict()) is dict


def test_config_with_list():
    dict_data = {
        'param1': [
            'value1',
            {
                'param1_1': 'value1_1',
                'param2_1': 'value2_1'
            }
        ]
    }
    config = Config(dict_data)

    assert type(config.param1[0]) is str
    assert type(config.param1[1]) is Config
    assert config.param1[0] == 'value1'
    assert config.param1[1].param2_1 == 'value2_1'


def test_set_value():
    dict_data = {
        'param1': 'value1'
    }
    config = Config(dict_data)
    config['param2'] = 'value2'
    config.param3 = 'value3'

    assert config.param2 == 'value2'
    assert config.param3 == 'value3'
    assert config.to_dict()['param2'] == 'value2'
    assert config.to_dict()['param3'] == 'value3'
