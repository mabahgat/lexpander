from pathlib import Path

import pytest
from lexicons import LookUpLexicon, LexiconFormatError, Liwc2015, LabelMapper, Values, LookUpLexiconWithMapping, Liwc22, \
    get_lexicon, InvalidLexiconName, ExpandedLexicon
from tests.utils import get_abs_file_path


def test_get_lexicon_liwc2015():
    liwc2015_path = __liwc2015_dic_path()
    liwc2015_params = {
        'dic_path': liwc2015_path,
        'strict_matching': False
    }
    liwc2015 = get_lexicon('liwc2015', **liwc2015_params)

    assert type(liwc2015) is Liwc2015
    assert liwc2015.get_conf()['file_path'] == str(liwc2015_path)
    assert liwc2015.get_conf()['strict'] == False


def test_get_lexicon_values():
    values_path = get_abs_file_path(__file__, 'resources/lexicons/test_values.csv')
    values_params = {
        'csv_path': values_path
    }
    values = get_lexicon('values', **values_params)

    assert type(values) is Values
    assert values.get_conf()['file_path'] == str(values_path)


def test_get_lexicon_liwc22():
    liwc22_path = get_abs_file_path(__file__, 'resources/lexicons/test_liwc22_lookup.csv')
    liwc22_params = {
        'csv_path': liwc22_path
    }
    liwc22 = get_lexicon('liwc22', **liwc22_params)

    assert type(liwc22) is Liwc22
    assert liwc22.get_conf()['file_path'] == str(liwc22_path)


def test_get_lexicon_custom():
    rand_lex_path = get_abs_file_path(__file__, 'resources/lexicons/rand_3_labels_150_examples.csv')
    rand_params = {
        'csv_path': rand_lex_path
    }
    rand_lex = get_lexicon(**rand_params)

    assert type(rand_lex) is LookUpLexicon
    assert rand_lex.get_conf()['file_path'] == str(rand_lex_path)


def test_get_lexicon_invalid_name():
    with pytest.raises(InvalidLexiconName):
        get_lexicon('blah')


def test_get_lexicon_bad_arguments():
    with pytest.raises(ValueError):
        get_lexicon()


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
    assert lex.label_term('word') == ['noun']
    assert sorted(lex.label_term('book')) == sorted(['noun', 'verb'])

    lex_conf = lex.get_conf()
    assert lex_conf['file_path'] == str(lex_path)
    assert lex_conf['sep'] == ','
    assert sorted(lex_conf['labels']) == sorted(['noun', 'verb', 'article'])
    assert lex_conf['label_count'] == 3
    assert lex_conf['entries_count'] == 7


def test_lookup_lexicon_incorrect_errors():
    with pytest.raises(FileNotFoundError):
        LookUpLexicon(Path('/path/not/exists'))
    with pytest.raises(LexiconFormatError):
        LookUpLexicon(get_abs_file_path(__file__, 'resources/lexicons/bad_lookup_lexicon.csv'))


def __liwc2015_dic_path():
    return get_abs_file_path(__file__, 'resources/lexicons/test_liwc2015.dic')


def test_liwc2015_config():
    dic_path = __liwc2015_dic_path()
    liwc_strict = Liwc2015(dic_path=dic_path, strict_matching=False)
    conf_strict = liwc_strict.get_conf()

    assert conf_strict['file_path'] == str(dic_path)
    assert conf_strict['strict'] == False
    assert conf_strict['labels'] == {'label1', 'label2', 'label3', 'label4', 'label5', 'label6'}
    assert conf_strict['label_count'] == 6

    liwc = Liwc2015(dic_path=dic_path)
    conf = liwc.get_conf()

    assert conf['strict'] == True
    assert conf['name'] == 'liwc2015'


def test_liwc2015_strict():
    liwc = Liwc2015(dic_path=__liwc2015_dic_path(), strict_matching=True)

    assert liwc.label_term('word1') == ['label1']
    assert liwc.label_term('blah') == []
    assert liwc.label_term('wild1') == ['label3', 'label5']
    assert liwc.label_term('wild1suffix') == []


def test_liwc2015():
    liwc = Liwc2015(dic_path=__liwc2015_dic_path(), strict_matching=False)

    assert liwc.label_term('word1suffix') == []
    assert liwc.label_term('wild1') == ['label3', 'label5']
    assert liwc.label_term('wild1suffix') == ['label3', 'label5']


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

    assert liwc.label_term('word1') == ['label1']
    assert liwc.label_term('word2') == ['label1', 'label3']
    assert liwc.label_term('word5') == []
    assert liwc.label_term('wild1') == ['label3']

    assert liwc.get_labels() == {'label1', 'label3'}
    assert conf['labels'] == {'label1', 'label3'}
    assert conf['label_count'] == 2


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

    assert lexicon.label_term('word1') == ['value1']
    assert lexicon.label_term('word3') == ['value3']
    assert lexicon.label_term('word4') == []
    assert lexicon.label_term('word5') == ['value1']
    assert lexicon.label_term('word6') == []

    assert conf['labels'] == {'value1', 'value3'}
    assert conf['label_count'] == 2
    assert 'label_map' in conf


def test_values():
    dict_path = get_abs_file_path(__file__, 'resources/lexicons/test_values.csv')
    lexicon = Values(csv_path=dict_path)
    conf = lexicon.get_conf()

    assert lexicon.label_term('word1') == ['value1', 'value2']
    assert lexicon.label_term('word4') == ['value2']

    assert conf['name'] == 'values'


def test_liwc2022_from_csv():
    path = get_abs_file_path(__file__, 'resources/lexicons/test_liwc22_lookup.csv')
    lexicon = Liwc22(path)
    conf = lexicon.get_conf()

    assert lexicon.label_term('pies') == ['Physical', 'food']
    assert lexicon.label_term('Pies') == ['Physical', 'food']
    assert lexicon.label_term('minute') == ['quantity', 'time']
    assert lexicon.label_term('minutes') == []
    assert conf['file_path'] == str(path)


def test_liwc2022_from_liwc_output():
    path = get_abs_file_path(__file__, 'resources/lexicons/test_liwc22_tool_output.csv')
    lexicon = Liwc22(path, from_tool_output=True)
    conf = lexicon.get_conf()

    assert lexicon.label_term('pies') == ['Physical', 'food']
    assert lexicon.label_term('Pies') == ['Physical', 'food']
    assert lexicon.label_term('swop') == []
    assert lexicon.label_term('Pieses') == []
    assert conf['file_path'] == str(path)


def test_liwc22_only_labels():
    path = get_abs_file_path(__file__, 'resources/lexicons/test_liwc22_lookup.csv')
    use_labels = {'Physical', 'quantity', 'time', 'Drives'}
    lexicon = Liwc22(path, use_labels=use_labels)
    conf = lexicon.get_conf()

    assert lexicon.label_term('pies') == ['Physical']
    assert lexicon.label_term('Pies') == ['Physical']
    assert lexicon.label_term('minute') == ['quantity', 'time']
    assert lexicon.label_term('book') == []
    assert lexicon.label_term('abusion') == ['Drives']
    assert lexicon.label_term('minutes') == []
    assert conf['file_path'] == str(path)


def test_expanded_lexicon():
    lex_path = get_abs_file_path(__file__, 'resources/lexicons/test_lookup_lexicon.csv')
    lex = LookUpLexicon(lex_path)

    new_lex = {
        'play': 'verb',
        'game': 'noun'
    }
    exp_lex = ExpandedLexicon('test_expanded', new_lex, lex)

    assert exp_lex.label_term('play') == ['verb']
    assert exp_lex.label_term('run') == ['verb']
    assert exp_lex.label_term('blah') == []


def test_expanded_lexicon_conf():
    lex_path = get_abs_file_path(__file__, 'resources/lexicons/test_lookup_lexicon.csv')
    lex = LookUpLexicon(lex_path)

    new_lex = {
        'play': 'verb',
        'game': 'noun'
    }
    exp_lex = ExpandedLexicon('test_expanded', new_lex, lex)
    exp_conf = exp_lex.get_conf()

    assert exp_conf['new_terms_count'] == 2


def test_expanded_lexicon_save(tmp_path):
    out_path = tmp_path

    lex_path = get_abs_file_path(__file__, 'resources/lexicons/test_lookup_lexicon.csv')
    lex = LookUpLexicon(lex_path)

    new_lex = {
        'play': 'verb',
        'game': 'noun'
    }
    exp_lex = ExpandedLexicon('test_expanded', new_lex, lex, out_path)
    exp_lex.save()

    csv_path = out_path / 'test_expanded__expanded_lex.csv'
    loaded_lex = LookUpLexicon(csv_path)

    assert loaded_lex.label_term('play') == ['verb']
    assert loaded_lex.label_term('game') == ['noun']
    assert loaded_lex.label_term('blah') == []
