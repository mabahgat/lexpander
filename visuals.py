import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_precision_recall_at(df: pd.DataFrame, thr: float) -> (float, float):
	"""
	Computes precision and recall at a specific threshold
	:param df: Dataframe with "label" as truth column and "label_out" and "prob_out" as model output
		probabilities are expected to be between 0 and 1
	:param thr: float value
	:return: tuple of two floats; precision and recall
	"""
	thr_df = df[df.prob_out >= thr]
	if len(thr_df) <= 0:
		return 0, 0
	correct_df = thr_df[thr_df.label == thr_df.label_out]
	precision = len(correct_df) / len(thr_df)
	recall = len(thr_df) / len(df)
	return precision, recall


def plot_precision_recall_curve(df: pd.DataFrame,
							    title: str = None,
							    font_size: int = 25,
							    line_width: int = 6,
							    x_lim: (float, float) = None,
							    y_lim: (float, float) = None):
	"""
	Plots precision and recall against list of thresholds (by default from 0 to 1 with 100 steps)
	"""
	thr_lst = [p/100 for p in range(0, 100, 1)]
	p_r_lst = [get_precision_recall_at(df, thr) for thr in thr_lst]
	p_lst, r_lst = list(zip(*p_r_lst))

	_, ax = plt.subplots()

	def set_minor_ticks(x_start, x_end):
		draw_range = np.arange(x_start, x_end + 0.1, step=0.05)
		ax.set_xticks(draw_range, minor=True)
		ax.set_yticks(draw_range, minor=True)

	start = np.amin(thr_lst)
	end = np.amax(thr_lst)
	set_minor_ticks(start, end)

	ax.plot(thr_lst, p_lst, label='Precision', linewidth=line_width)
	ax.plot(thr_lst, r_lst, label='Recall', linewidth=line_width)
	ax.set_xlabel('Threshold', fontsize=font_size)
	ax.set_ylabel('Precision/Recall', fontsize=font_size)
	if x_lim:
		ax.set_xlim(x_lim)
	if y_lim:
		ax.set_ylim(y_lim)
	ax.grid(which='major')
	ax.grid(which='minor', linestyle='--')
	if title:
		ax.set_title(title, fontsize=font_size)
	ax.legend()
	plt.xticks(fontsize=font_size, rotation=90)
	plt.yticks(fontsize=font_size)

	plt.show()

	return ax
