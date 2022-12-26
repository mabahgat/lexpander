import pytest
from lexicons import LookUpLexicon, LexiconFormatError, Liwc2015, LabelMapper, Values, LookUpLexiconWithMapping, Liwc22
from tests.utils import get_abs_file_path


def test_label_mapper():
    label_map = {
        'label1': None,
        'label2': 'label1',
        'label3': 'label1',
        'label4': 'label3',
        'label5': 'label3',
        'label6': None,
        'label7': 'label5'
    }
    used_labels = {'label1', 'label3'}
    mapper = LabelMapper(use_labels=used_labels, label_map=label_map)

    assert mapper.map('label2') == 'label1'
    assert mapper.map('label7') == 'label3'

    assert mapper.map_list(['label1', 'label2']) == ['label1']

    assert mapper.get_conf()


def test_lookup_lexicon():
    lex_path = get_abs_file_path(__file__, 'resources/lexicons/test_lookup_lexicon.csv')
    lex = LookUpLexicon(lex_path)

    assert sorted(lex.get_labels()) == sorted(['noun', 'verb', 'article'])
    assert lex.label_token('word') == ['noun']
    assert sorted(lex.label_token('book')) == sorted(['noun', 'verb'])

    lex_conf = lex.get_conf()
    assert lex_path == lex_conf['file_path']
    assert lex_conf['sep'] == ','
    assert sorted(lex_conf['labels']) == sorted(['noun', 'verb', 'article'])
    assert lex_conf['label_count'] == 3
    assert lex_conf['entries_count'] == 7


def test_lookup_lexicon_incorrect_errors():
    with pytest.raises(FileNotFoundError):
        LookUpLexicon('blah')
    with pytest.raises(LexiconFormatError):
        LookUpLexicon(get_abs_file_path(__file__, 'resources/lexicons/bad_lookup_lexicon.csv'))


def __liwc2015_dic_path():
    return get_abs_file_path(__file__, 'resources/lexicons/test_liwc2015.dic')


def test_liwc2015_config():
    dic_path = __liwc2015_dic_path()
    liwc_strict = Liwc2015(dic_path=dic_path, strict_matching=True)
    conf_strict = liwc_strict.get_conf()

    assert conf_strict['file_path'] == dic_path
    assert conf_strict['strict'] == True
    assert conf_strict['labels'] == {'label1', 'label2', 'label3', 'label4', 'label5', 'label6'}
    assert conf_strict['label_count'] == 6

    liwc = Liwc2015(dic_path=dic_path)
    conf = liwc.get_conf()

    assert conf['strict'] == False
    assert conf['name'] == 'liwc2015'


def test_liwc2015_strict():
    liwc = Liwc2015(dic_path=__liwc2015_dic_path(), strict_matching=True)

    assert liwc.label_token('word1') == ['label1']
    assert liwc.label_token('blah') == []
    assert liwc.label_token('wild1') == ['label3', 'label5']
    assert liwc.label_token('wild1suffix') == []


def test_liwc2015():
    liwc = Liwc2015(dic_path=__liwc2015_dic_path())

    assert liwc.label_token('word1suffix') == []
    assert liwc.label_token('wild1') == ['label3', 'label5']
    assert liwc.label_token('wild1suffix') == ['label3', 'label5']


def test_liwc2015_limited_cats():
    label_map = {
        'label1': None,
        'label2': 'label1',
        'label3': 'label1',
        'label4': 'label3',
        'label5': 'label3',
        'label6': None
    }
    liwc = Liwc2015(use_labels={'label1', 'label3'},
                    label_map=label_map,
                    dic_path=__liwc2015_dic_path())
    conf = liwc.get_conf()

    assert liwc.label_token('word1') == ['label1']
    assert liwc.label_token('word2') == ['label1', 'label3']
    assert liwc.label_token('word5') == []
    assert liwc.label_token('wild1') == ['label3']

    assert liwc.get_labels() == {'label1', 'label3'}
    assert conf['labels'] == {'label1', 'label3'}
    assert conf['label_count'] == 2


def test_liwc2015_load_default_path():
    lexicon = Liwc2015()
    conf = lexicon.get_conf()
    assert conf['file_path'] is not None


def test_lookup_lexicon_with_mapping():
    dict_path = get_abs_file_path(__file__, 'resources/lexicons/test_values.csv')
    used_labels = {'value1', 'value3'}
    label_map = {
        'value1': None,
        'value2': None,
        'value3': None,
        'value4': 'value2',
        'value5': 'value1',
        'value6': 'value2'
    }
    lexicon = LookUpLexiconWithMapping(csv_path=dict_path,
                                       use_labels=used_labels,
                                       label_map=label_map)
    conf = lexicon.get_conf()

    assert lexicon.label_token('word1') == ['value1']
    assert lexicon.label_token('word3') == ['value3']
    assert lexicon.label_token('word4') == []
    assert lexicon.label_token('word5') == ['value1']
    assert lexicon.label_token('word6') == []

    assert conf['labels'] == {'value1', 'value3'}
    assert conf['label_count'] == 2
    assert 'label_map' in conf


def test_values():
    dict_path = get_abs_file_path(__file__, 'resources/lexicons/test_values.csv')
    lexicon = Values(csv_path=dict_path)
    conf = lexicon.get_conf()

    assert lexicon.label_token('word1') == ['value1', 'value2']
    assert lexicon.label_token('word4') == ['value2']

    assert conf['name'] == 'values'


def test_values_load_default_path():
    lexicon = Values()
    conf = lexicon.get_conf()
    assert conf['file_path'] is not None


def test_liwc2022_from_csv():
    path = get_abs_file_path(__file__, 'resources/lexicons/test_liwc22_lookup.csv')
    lexicon = Liwc22(path)
    conf = lexicon.get_conf()

    assert lexicon.label_token('pies') == ['Physical', 'food']
    assert lexicon.label_token('Pies') == ['Physical', 'food']
    assert lexicon.label_token('minute') == ['quantity', 'time']
    assert lexicon.label_token('minutes') == []
    assert conf['file_path'] == path


def test_liwc2022_from_liwc_output():
    path = get_abs_file_path(__file__, 'resources/lexicons/test_liwc22_tool_output.csv')
    lexicon = Liwc22(path, from_tool_output=True)
    conf = lexicon.get_conf()

    assert lexicon.label_token('pies') == ['Physical', 'food']
    assert lexicon.label_token('Pies') == ['Physical', 'food']
    assert lexicon.label_token('swop') == []
    assert lexicon.label_token('Pieses') == []
    assert conf['file_path'] == path


def test_liwc22_only_labels():
    path = get_abs_file_path(__file__, 'resources/lexicons/test_liwc22_lookup.csv')
    use_labels = {'Physical', 'quantity', 'time', 'Drives'}
    lexicon = Liwc22(path, use_labels=use_labels)
    conf = lexicon.get_conf()

    assert lexicon.label_token('pies') == ['Physical']
    assert lexicon.label_token('Pies') == ['Physical']
    assert lexicon.label_token('minute') == ['quantity', 'time']
    assert lexicon.label_token('book') == []
    assert lexicon.label_token('abusion') == ['Drives']
    assert lexicon.label_token('minutes') == []
    assert conf['file_path'] == path


def test_liwc22_load_default_path():
    lexicon = Liwc22(from_tool_output=True)
    conf = lexicon.get_conf()
    assert conf['file_path'] is not None
