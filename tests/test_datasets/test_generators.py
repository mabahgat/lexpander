from datasets.base import list_dataset_names
from datasets.generators import DatasetGenerator
from lexicons import LookUpLexicon
from dictionaries import SimpleDictionary, Dictionary
from tests.utils import get_abs_file_path


def __get_test_lexicon():
    lexicon_path = get_abs_file_path(__file__, '../resources/lexicons/rand_lex_5_labels_words_0-349.csv')
    return LookUpLexicon(lexicon_path)


def __get_test_dictionary_1():
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
    dict_path = get_abs_file_path(__file__, '../resources/dictionaries/rand_dict_5_labels_words_0-299.csv')
    return SimpleDictionary(name='test-1', file_path=dict_path)


def __get_test_dictionary_2():
    """
    Label Distribution for this file
    label_1	61
    label_2	58
    label_3	55
    label_4	51
    label_5	75
    Average quality - 4.04667
    :return:
    """
    dict_path = get_abs_file_path(__file__, '../resources/dictionaries/rand_dict_5_labels_words_50-349.csv')
    return SimpleDictionary(name='test-2', file_path=dict_path)


def __get_test_shuffled_dictionary():
    dict_path = get_abs_file_path(__file__, '../resources/dictionaries/rand_5_labels_0-299_shuffled.csv')
    return SimpleDictionary(name='test-2', file_path=dict_path)


def test_dataset_generator_conf():
    lexicon = __get_test_lexicon()
    dictionary = __get_test_dictionary_1()
    generator = DatasetGenerator(exp_name='some_exp',
                                 lexicon=lexicon,
                                 dictionaries=dictionary,
                                 test_count=30,
                                 exclusions=[])
    conf = generator.get_conf()
    assert conf['name'] == 'some_exp'
    assert conf['lexicon'] == lexicon.get_conf()
    assert len(conf['dictionaries']) == 1
    assert conf['dictionaries'][0] == dictionary.get_conf()
    assert conf['user_test_count'] == 30
    assert conf['force_test_count'] == False
    assert conf['test_percentage'] is None
    assert conf['same_train_set'] == False
    assert conf['exclusions'] == []


def test_generate_single_test_count(tmp_path):
    lexicon = __get_test_lexicon()
    dictionary = __get_test_dictionary_1()
    generator = DatasetGenerator(exp_name='exp_count',
                                 lexicon=lexicon,
                                 dictionaries=dictionary,
                                 test_count=30,
                                 force_test_count=True,
                                 exclusions=[],
                                 dataset_root_path=tmp_path)
    dataset = generator.generate()[0]
    assert len(dataset.get_test()) == 30


def test_generate_single_dataset_files(tmp_path):
    datasets_root_path = tmp_path
    lexicon = __get_test_lexicon()
    dictionary = __get_test_dictionary_1()
    generator = DatasetGenerator(exp_name='exp_datasets',
                                 lexicon=lexicon,
                                 dictionaries=dictionary,
                                 test_count=30,
                                 exclusions=[],
                                 dataset_root_path=datasets_root_path)
    generator.generate()

    assert 'exp_datasets_test-1' in list_dataset_names(datasets_root_path)
    assert (datasets_root_path / 'exp_datasets_test-1' / 'exp_datasets_test-1__train.csv').exists()
    assert (datasets_root_path / 'exp_datasets_test-1' / 'exp_datasets_test-1__test.csv').exists()
    assert (datasets_root_path / 'exp_datasets_test-1' / 'exp_datasets_test-1__conf.yaml').exists()


def test_generate_single_test_train_mutually_exclusive(tmp_path):
    lexicon = __get_test_lexicon()
    dictionary = __get_test_dictionary_1()
    generator = DatasetGenerator(exp_name='exp_test',
                                 lexicon=lexicon,
                                 dictionaries=dictionary,
                                 test_count=30,
                                 exclusions=[],
                                 dataset_root_path=tmp_path)
    dataset = generator.generate()[0]
    train_df = dataset.get_train()
    test_df = dataset.get_test()
    train_words_set = set(train_df.word.to_list())
    test_words_set = set(test_df.word.to_list())
    assert len(train_words_set & test_words_set) == 0


def test_generate_with_percentage(tmp_path):
    lexicon = __get_test_lexicon()
    dictionary = __get_test_dictionary_1()
    generator = DatasetGenerator(exp_name='exp_percentage',
                                 lexicon=lexicon,
                                 dictionaries=dictionary,
                                 test_percentage=0.1,
                                 exclusions=[],
                                 dataset_root_path=tmp_path)
    dataset = generator.generate()[0]
    assert len(dataset.get_test()) == 29


def test_generate_with_percentage_and_forced_count(tmp_path):
    lexicon = __get_test_lexicon()
    dictionary = __get_test_dictionary_1()
    generator = DatasetGenerator(exp_name='exp_percentage',
                                 lexicon=lexicon,
                                 dictionaries=dictionary,
                                 test_percentage=0.1,
                                 force_test_count=True,
                                 exclusions=[],
                                 dataset_root_path=tmp_path)
    dataset = generator.generate()[0]
    assert len(dataset.get_test()) == 30


def test_generate_for_quality(tmp_path):
    lexicon = __get_test_lexicon()
    dictionary = __get_test_dictionary_1()
    generator = DatasetGenerator(exp_name='exp_percentage',
                                 lexicon=lexicon,
                                 dictionaries=dictionary,
                                 test_percentage=0.1,
                                 quality_threshold=2,
                                 exclusions=[],
                                 dataset_root_path=tmp_path)
    dataset = generator.generate()[0]
    all_df = dataset.get_all()
    assert len(all_df[all_df.quality < 2]) == 0


def test_generate_test_label_distribution(tmp_path):
    lexicon = __get_test_lexicon()
    dictionary = __get_test_dictionary_1()
    generator = DatasetGenerator(exp_name='exp_percentage',
                                 lexicon=lexicon,
                                 dictionaries=dictionary,
                                 test_percentage=0.1,
                                 exclusions=[],
                                 dataset_root_path=tmp_path)
    dataset = generator.generate()[0]
    label_counts = dataset.get_test().label.value_counts().to_dict()
    assert len(dataset.get_test()) == 29
    assert label_counts['label_1'] == 6
    assert label_counts['label_2'] == 6
    assert label_counts['label_3'] == 5
    assert label_counts['label_4'] == 5
    assert label_counts['label_5'] == 7


def test_generate_test_label_distribution_with_force_test_count(tmp_path):
    lexicon = __get_test_lexicon()
    dictionary = __get_test_dictionary_1()
    generator = DatasetGenerator(exp_name='exp_percentage',
                                 lexicon=lexicon,
                                 dictionaries=dictionary,
                                 test_percentage=0.1,
                                 force_test_count=True,
                                 exclusions=[],
                                 dataset_root_path=tmp_path)
    dataset = generator.generate()[0]
    label_counts = dataset.get_test().label.value_counts().to_dict()
    assert len(dataset.get_test()) == 30
    assert label_counts['label_1'] == 6 or label_counts['label_1'] == 7
    assert label_counts['label_2'] == 6 or label_counts['label_2'] == 7
    assert label_counts['label_3'] == 5 or label_counts['label_3'] == 6
    assert label_counts['label_4'] == 5 or label_counts['label_4'] == 6
    assert label_counts['label_5'] == 7 or label_counts['label_5'] == 8


def test_generate_test_highest_quality(tmp_path):
    lexicon = __get_test_lexicon()
    dictionary = __get_test_shuffled_dictionary()
    generator = DatasetGenerator(exp_name='exp_percentage',
                                 lexicon=lexicon,
                                 dictionaries=dictionary,
                                 test_percentage=0.1,
                                 force_test_count=True,
                                 exclusions=[],
                                 dataset_root_path=tmp_path)
    dataset = generator.generate()[0]
    all_df = dictionary.get_all_records()
    test_df = dataset.get_test()

    def check_highest_quality(row):
        test_term = row[Dictionary.WORD_COLUMN]
        test_quality = row[Dictionary.QUALITY_COLUMN]
        highest_quality = all_df[all_df.word == test_term].quality.sort_values(ascending=False).head(1).to_list()[0]
        assert test_quality == highest_quality
    test_df.apply(check_highest_quality, axis=1)


def test_generate_with_two(tmp_path):
    lexicon = __get_test_lexicon()
    dictionary1 = __get_test_dictionary_1()
    dictionary2 = __get_test_dictionary_2()
    generator = DatasetGenerator(exp_name='exp_two_datasets',
                                 lexicon=lexicon,
                                 dictionaries=[dictionary1, dictionary2],
                                 test_count=30,
                                 force_test_count=True,
                                 exclusions=[],
                                 dataset_root_path=tmp_path)
    dataset1, dataset2 = generator.generate()

    assert len(dataset1.get_test()) == 30
    assert len(dataset2.get_test()) == 30

    test_terms_1 = set(dataset1.get_test()['word'].to_list())
    test_terms_2 = set(dataset2.get_test()['word'].to_list())
    assert test_terms_1 == test_terms_2


def test_generate_with_two_files(tmp_path):
    datasets_root_path = tmp_path

    lexicon = __get_test_lexicon()
    dictionary1 = __get_test_dictionary_1()
    dictionary2 = __get_test_dictionary_2()
    generator = DatasetGenerator(exp_name='exp_two_datasets_stored',
                                 lexicon=lexicon,
                                 dictionaries=[dictionary1, dictionary2],
                                 test_count=30,
                                 force_test_count=True,
                                 exclusions=[],
                                 dataset_root_path=tmp_path)
    generator.generate()

    assert 'exp_two_datasets_stored_test-1' in list_dataset_names(datasets_root_path)
    assert (datasets_root_path /
            'exp_two_datasets_stored_test-1' / 'exp_two_datasets_stored_test-1__train.csv').exists()
    assert (datasets_root_path /
            'exp_two_datasets_stored_test-1' / 'exp_two_datasets_stored_test-1__test.csv').exists()
    assert (datasets_root_path /
            'exp_two_datasets_stored_test-1' / 'exp_two_datasets_stored_test-1__conf.yaml').exists()

    assert 'exp_two_datasets_stored_test-2' in list_dataset_names(datasets_root_path)
    assert (datasets_root_path /
            'exp_two_datasets_stored_test-2' / 'exp_two_datasets_stored_test-2__train.csv').exists()
    assert (datasets_root_path /
            'exp_two_datasets_stored_test-2' / 'exp_two_datasets_stored_test-2__test.csv').exists()
    assert (datasets_root_path /
            'exp_two_datasets_stored_test-2' / 'exp_two_datasets_stored_test-2__conf.yaml').exists()
