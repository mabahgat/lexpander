import pandas as pd
import pytest

from dictionaries import UrbanDictionary, Wiktionary, get_dictionary_by_name, InvalidDictionaryName
from tests.utils import get_abs_file_path


def test_get_dictionary_by_name():
    ud_path = get_abs_file_path(__file__, 'resources/dictionaries/ud.csv')
    ud = get_dictionary_by_name('ud', custom_path=ud_path)
    assert type(ud) is UrbanDictionary

    ud2 = get_dictionary_by_name('urban_dictionary', custom_path=ud_path)
    assert type(ud2) is UrbanDictionary

    wk_path = get_abs_file_path(__file__, 'resources/dictionaries/wiktionary_raw.csv')
    wk = get_dictionary_by_name('wiktionary', custom_path=wk_path)
    assert type(wk) is Wiktionary

    with pytest.raises(InvalidDictionaryName):
        get_dictionary_by_name('blah')


def test_urban_dictionary_load_raw():
    file_path = get_abs_file_path(__file__, 'resources/dictionaries/ud.dat')
    dictionary = UrbanDictionary(file_path=file_path, file_type='raw')
    conf = dictionary.get_conf()
    records = dictionary.get_all_records()

    assert len(records) == 13
    assert records.columns.to_list() == ['word', 'text', 'quality']
    assert records.iloc[0].word == 'b stalking'  # test text correction
    assert records.iloc[3].word == 'dhet'   # case check
    assert records[records.word == 'bad entry'].empty   # remove bad entries

    assert conf['name'] == 'UrbanDictionary'
    assert conf['count'] == 13
    assert conf['file_path'] == file_path
    assert conf['file_type'] == 'raw'


def test_urban_dictionary_load_csv():
    file_path = get_abs_file_path(__file__, 'resources/dictionaries/ud.csv')
    dictionary = UrbanDictionary(file_path=file_path, file_type='csv')
    records = dictionary.get_all_records()

    assert len(records) == 2


def test_urban_dictionary_load_raw_text_check():
    file_path = get_abs_file_path(__file__, 'resources/dictionaries/ud.dat')
    dictionary = UrbanDictionary(file_path=file_path, file_type='raw')
    records = dictionary.get_all_records()

    assert records.iloc[12].word == 'case test'
    assert records.iloc[12].text == 'tag list Some Meaning Some Example'


def test_urban_dictionary_load_raw_save_csv(tmp_path):
    dst_path = tmp_path / 'ud.csv'
    file_path = get_abs_file_path(__file__, 'resources/dictionaries/ud.dat')
    dictionary = UrbanDictionary(file_path=file_path, file_type='raw')
    dictionary.save_as_csv(str(dst_path))

    assert dst_path.exists()

    saved_df = pd.read_csv(dst_path, index_col=0)
    assert saved_df.columns.to_list() == ['word', 'text', 'quality']
    assert saved_df.iloc[3].word == 'dhet'


def test_urban_dictionary_get_records_copy(tmp_path):
    dst_path = tmp_path / 'ud.csv'
    file_path = get_abs_file_path(__file__, 'resources/dictionaries/ud.dat')
    dictionary = UrbanDictionary(file_path=file_path, file_type='raw')

    records_copy = dictionary.get_all_records()
    records_copy.word = "new word"
    records = dictionary.get_all_records()
    assert records.iloc[0].word != "new word"


def test_wiktionary_load_raw():
    file_path = get_abs_file_path(__file__, 'resources/dictionaries/wiktionary_raw.csv')
    dictionary = Wiktionary(file_path=file_path, file_type='raw')
    conf = dictionary.get_conf()
    records = dictionary.get_all_records()

    assert len(records) == 15
    assert records.columns.to_list() == ['word', 'text', 'quality']
    assert records.iloc[0].word == 'gnu fdl'
    assert records.iloc[10].word == 'livre'   # case check

    assert conf['name'] == 'Wiktionary'
    assert conf['count'] == 15
    assert conf['file_path'] == file_path
    assert conf['file_type'] == 'raw'


def test_wiktionary_load_raw_empty_checks():
    file_path = get_abs_file_path(__file__, 'resources/dictionaries/wiktionary_raw.csv')
    dictionary = Wiktionary(file_path=file_path, file_type='raw')
    records = dictionary.get_all_records()

    assert len(records) == 15
    assert records.iloc[11].text == 'meaning with, comma example with, comma'
    assert records.iloc[12].text == 'meaning only'
    assert records.iloc[13].text == 'example only'
    assert records.iloc[14].word == 'entry 3'
