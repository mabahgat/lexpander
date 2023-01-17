import numpy as np
import pandas as pd
import pytest

from visuals import get_precision_recall_at, plot_precision_recall_curve, __get_thresholds_list, \
	plot_counts_vs_threshold


def test_get_precision_recall_at():
	data = {
		'word': ['word1', 'word2', 'word3', 'word4'],
		'label': ['label1', 'label2', 'label3', 'label4'],
		'label_out': ['label1', 'label2', 'label3', 'label4'],
		'prob_out': [0.1, 0.4, 0.7, 0.9]
	}
	df = pd.DataFrame.from_dict(data)

	p, r = get_precision_recall_at(df, 0)
	assert p == 1
	assert r == 1

	p, r = get_precision_recall_at(df, 1)
	assert p == 0
	assert r == 0

	p, r = get_precision_recall_at(df, 0.5)
	assert p == 1
	assert r == 0.5

	data2 = {
		'word': ['word1', 'word2', 'word3', 'word4'],
		'label': ['label1', 'label2', 'label3', 'label4'],
		'label_out': ['label1', 'label2', 'label2', 'label3'],
		'prob_out': [0.4, 0.6, 0.7, 0.9]
	}
	df2 = pd.DataFrame.from_dict(data2)

	p, r = get_precision_recall_at(df2, 0)
	assert p == 0.5
	assert r == 1

	p, r = get_precision_recall_at(df2, 0.5)
	assert p == pytest.approx(0.333, 0.01)
	assert r == 0.75


def test_get_threshold_list():
	thr_lst = __get_thresholds_list()

	assert np.amin(thr_lst) == 0
	assert np.amax(thr_lst) == 1
	assert len(thr_lst) == 101


def test_plot_precision_recall_curve():
	data = {
		'word': ['word1', 'word2', 'word3', 'word4'],
		'label': ['label1', 'label2', 'label3', 'label4'],
		'label_out': ['label1', 'label2', 'label3', 'label4'],
		'prob_out': [0.1, 0.4, 0.7, 0.9]
	}
	df = pd.DataFrame.from_dict(data)
	tpr = plot_precision_recall_curve(df, title='curve', x_lim=(0, 1), y_lim=(0, 1))

	assert tpr[0] == (1, 1)
	assert tpr[0.5] == (1, 0.5)
	assert tpr[1] == (0, 0)


def test_plot_counts_vs_threshold():
	data = {
		'word': ['word1', 'word2', 'word3', 'word4'],
		'label': ['label1', 'label2', 'label3', 'label4'],
		'label_out': ['label1', 'label2', 'label3', 'label4'],
		'prob_out': [0.1, 0.4, 0.7, 0.9]
	}
	df = pd.DataFrame.from_dict(data)
	counts = plot_counts_vs_threshold(df)

	thr_count_dict = {thr: count for thr, count in zip(counts['threshold'], counts['count'])}
	assert thr_count_dict[0] == 4
	assert thr_count_dict[0.5] == 2
	assert thr_count_dict[1] == 0
