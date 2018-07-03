import matplotlib.pyplot as plt
from collections import Counter
from missingno import matrix, heatmap, dendrogram
from seaborn import set, boxplot, distplot, despine, pairplot
from sklearn import decomposition

def hist(series):
	try:
		label = series.name.capitalize()
		fig = plt.figure()
		plt.hist(series, 20, normed = 1, facecolor = 'blue', alpha = 0.75)
		plt.xlabel(label)
		plt.ylabel('Probability')
		plt.title('Histogram of ' + label)
		fig.savefig('datascience/' + label + '_hist.png')
		plt.close(fig)
	except:
		pass

def boxplot(series):
	try:
		fig = plt.figure()
		plt.boxplot(series, 0, 'rs', 0)
		label = series.name.capitalize()
		plt.title('Boxplot of ' + label)
		fig.savefig('datascience/' + label + '_box.png')
		plt.close(fig)
	except:
		pass

def counts_bargraph(series):
	try:
		fig = plt.figure()
		counts = Counter(series)
		length = range(len(counts))
		values = list(counts.values())
		keys = list(counts.keys())
		label = series.name.capitalize()
		plt.bar(length, values, align = 'center')
		plt.xticks(length, keys)
		plt.title('Bar graph of ' + label)
		fig.savefig('datascience/' + label + '_bar.png')
		plt.close(fig)
	except:
		pass

def missing_df(df):
	try:
		fig = plt.figure()
		mat = matrix(df)
		ax = plt.gca()
		plt.savefig('datascience/' + 'missing_df.png')
		plt.close(fig)
	except:
		pass

def missing_heatmap(df):
	try:
		fig = plt.figure()
		hm = heatmap(df)
		ax = plt.gca()
		plt.savefig('datascience/' + 'missing_heatmap.png')
		plt.close(fig)
	except:
		pass

def missing_dendrogram(df):
	try:
		fig = plt.figure()
		dg = dendrogram(df)
		ax = plt.gca()
		plt.savefig('datascience/' + 'missing_dendrogram.png')
		plt.close(fig)
	except:
		pass

def hist_with_boxplot_density(series):
	try:
		label = series.name.capitalize()
		fig = plt.figure()
		set(style = "ticks")
		f, (ax_box, ax_hist) = plt.subplots(2, sharex = True, gridspec_kw = {"height_ratios": (.15, .85)})
		boxplot(series, ax = ax_box)
		distplot(series, ax = ax_hist)
		ax_box.set(yticks = [])
		despine(ax = ax_hist)
		despine(ax=ax_box, left = True)
		ax = plt.gca()
		plt.savefig('datascience/' + label + '_hist_box_density.png')
		plt.close(fig)
	except:
		pass

def vis_corr_mat(corr_mat):
	try:
		fig = plt.figure()
		plt.matshow(corr_mat)
		plt.title('Correlation Matrix')
		ax = plt.gca()
		plt.savefig('datascience/' + 'corr_mat.png')
		plt.close(fig)
	except:
		pass

def pairwise_scatter(data, continuous = None):
	try:
		fig = plt.figure()
		if continuous is not None:
			data = data[continuous]

		set(style="ticks", color_codes=True)
		pairplot(data)
		ax = plt.gca()
		plt.savefig('datascience/' + 'scatter_matrix.png')
		plt.close(fig)
	except:
		pass

def plot_raw_data(data, corr_matrix, transforms):
	data[transforms["continuous"]].apply(hist)
	data[transforms["continuous"]].apply(boxplot)
	data[transforms["continuous"]].apply(hist_with_boxplot_density)
	data[transforms["categorical"]].apply(counts_bargraph)
	missing_df(data)
	missing_heatmap(data)
	missing_dendrogram(data)
	vis_corr_mat(corr_matrix)
	pairwise_scatter(data, transforms["continuous"])

def plot_processed_data(data):
	visualize_truncated_svd(data, transforms["continuous"])

def visualize_truncated_svd(data, continuous):
	try:
		tsvd = decomposition.TruncatedSVD()
		tsvd.fit(data[continuous])
		data2 = tsvd.transform(data[continuous])
		pairwise_scatter(data2)
	except:
		pass
