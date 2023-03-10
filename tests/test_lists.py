from lists import StopWordsLookUp, FileLookUp, NamesLookUp
from tests.utils import get_abs_file_path


def test_file_lookup():
    lookup_path = get_abs_file_path(__file__, 'resources/lists/names.lst')
    lookup = FileLookUp(lookup_path)
    conf = lookup.get_conf()

    assert lookup.contains('sara')
    assert not lookup.contains('barbra')

    assert conf['file_path'] == str(lookup_path)
    assert conf['count'] == 9
    assert conf['case_sensitive'] == False


def test_names_lookup():
    lookup_path = get_abs_file_path(__file__, 'resources/lists/names.lst')
    lookup = NamesLookUp(lookup_path)
    conf = lookup.get_conf()

    assert lookup.contains('sara')
    assert not lookup.contains('barbra')

    assert conf['file_path'] == str(lookup_path)
    assert conf['count'] == 9
    assert conf['case_sensitive'] == False
    assert conf['name'] == 'names'


def test_stopwords():
    stopwords = StopWordsLookUp()
    conf = stopwords.get_conf()

    assert stopwords.contains('a')
    assert conf['library'] is not None
    assert conf['case_sensitive'] == False
