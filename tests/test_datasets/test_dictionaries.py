import pandas as pd

from datasets.dictionaries import list_dataset_names, DictionaryDataset, DatasetNotFoundError
import pytest


def test_list_dataset_names(tmp_path):
    datasets_root_path = tmp_path
    (datasets_root_path / 'dataset1').mkdir()
    (datasets_root_path / 'dataset2').mkdir()
    (datasets_root_path / 'dataset3').mkdir()

    datasets = list_dataset_names(datasets_root_path)

    assert 'dataset1' in datasets
    assert 'dataset2' in datasets
    assert 'dataset3' in datasets


def test_dictionary_datasets_new(tmp_path):
    datasets_root_path = tmp_path

    data = {'words': ['me', 'you'], 'labels': ['self', 'other']}
    train_df = pd.DataFrame.from_dict(data)
    valid_df = pd.DataFrame.from_dict(data)
    test_df = pd.DataFrame.from_dict(data)

    dataset = DictionaryDataset(name='test_dataset',
                                train_df=train_df,
                                valid_df=valid_df,
                                test_df=test_df,
                                source_path='/some/path',
                                datasets_root_path=datasets_root_path)

    assert 'test_dataset' in list_dataset_names(datasets_root_path)
    assert (datasets_root_path / 'test_dataset' / 'test_dataset__train.csv').exists()
    assert (datasets_root_path / 'test_dataset' / 'test_dataset__valid.csv').exists()
    assert (datasets_root_path / 'test_dataset' / 'test_dataset__test.csv').exists()
    assert (datasets_root_path / 'test_dataset' / 'test_dataset__conf.yaml').exists()

    assert dataset.get_train() is not None
    assert dataset.get_valid() is not None
    assert dataset.get_test() is not None
    assert dataset.get_all() is not None
    assert len(dataset.get_all()) == 6

    conf = dataset.get_conf()
    assert conf['name'] == 'test_dataset'
    assert conf['root_path'] == str(datasets_root_path / 'test_dataset')
    assert conf['train_path'] == str(datasets_root_path / 'test_dataset' / 'test_dataset__train.csv')
    assert conf['valid_path'] == str(datasets_root_path / 'test_dataset' / 'test_dataset__valid.csv')
    assert conf['test_path'] == str(datasets_root_path / 'test_dataset' / 'test_dataset__test.csv')
    assert conf['train_count'] == 2
    assert conf['valid_count'] == 2
    assert conf['test_count'] == 2
    assert conf['source_file'] == '/some/path'

    copy_train_df = dataset.get_train()
    copy_train_df['labels'] = ['another_self', 'another_other']
    assert dataset.get_train()['labels'].to_list() == ['self', 'other']

    copy_valid_df = dataset.get_valid()
    copy_valid_df['labels'] = ['another_self', 'another_other']
    assert dataset.get_valid()['labels'].to_list() == ['self', 'other']

    copy_test_df = dataset.get_test()
    copy_test_df['labels'] = ['another_self', 'another_other']
    assert dataset.get_test()['labels'].to_list() == ['self', 'other']


def test_dictionary_dataset_load(tmp_path):
    datasets_root_path = tmp_path

    data = {'words': ['me', 'you'], 'labels': ['self', 'other']}
    train_df = pd.DataFrame.from_dict(data)
    valid_df = pd.DataFrame.from_dict(data)
    test_df = pd.DataFrame.from_dict(data)

    DictionaryDataset(name='test_dataset',
                      train_df=train_df,
                      valid_df=valid_df,
                      test_df=test_df,
                      source_path='/some/path',
                      datasets_root_path=datasets_root_path)

    new_dataset = DictionaryDataset(name='test_dataset',
                                    datasets_root_path=datasets_root_path)

    conf = new_dataset.get_conf()
    assert conf['name'] == 'test_dataset'
    assert conf['root_path'] == str(datasets_root_path / 'test_dataset')
    assert conf['train_path'] == str(datasets_root_path / 'test_dataset' / 'test_dataset__train.csv')
    assert conf['valid_path'] == str(datasets_root_path / 'test_dataset' / 'test_dataset__valid.csv')
    assert conf['test_path'] == str(datasets_root_path / 'test_dataset' / 'test_dataset__test.csv')
    assert conf['train_count'] == 2
    assert conf['valid_count'] == 2
    assert conf['test_count'] == 2
    assert conf['source_file'] == '/some/path'

    assert new_dataset.get_train() is not None
    assert new_dataset.get_valid() is not None
    assert new_dataset.get_test() is not None
    assert new_dataset.get_all() is not None
    assert len(new_dataset.get_all()) == 6


def test_dictionary_dataset_no_name_error():
    with pytest.raises(ValueError):
        DictionaryDataset(name=None)


def test_dictionary_dataset_not_found_error(tmp_path):
    datasets_root_path = tmp_path
    with pytest.raises(DatasetNotFoundError):
        DictionaryDataset('blah', datasets_root_path=datasets_root_path)
