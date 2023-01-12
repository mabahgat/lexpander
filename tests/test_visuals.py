import pandas as pd
import pytest

from visuals import get_precision_recall_at, plot_precision_recall_curve


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


def test_plot_precision_recall_curve():
	data = {
		'word': ['word1', 'word2', 'word3', 'word4'],
		'label': ['label1', 'label2', 'label3', 'label4'],
		'label_out': ['label1', 'label2', 'label3', 'label4'],
		'prob_out': [0.1, 0.4, 0.7, 0.9]
	}
	df = pd.DataFrame.from_dict(data)
	ax = plot_precision_recall_curve(df,
									 title='curve',
									 x_lim=(0, 1),
									 y_lim=(0, 1))

	assert ax.get_title() == 'curve'
	assert ax.get_xlim() == (0, 1)
	assert ax.get_ylim() == (0, 1)
