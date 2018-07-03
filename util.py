from pandas import concat, Series, DataFrame
from math import isnan, ceil
from numpy import NaN

def process_cluster_df(data1, cluster_col = "cluster", composite = False):
	clusters = data1[cluster_col]
	data_means = data1.groupby(cluster_col).mean()
	if not composite:
		data_means[cluster_col] = data_means.index.astype("int32")
	data_columns = data1.columns
	data1 = data1.join(data_means, on = cluster_col, how = "inner", lsuffix = '', rsuffix = '_mean')
	columns = [str(a) for a in data_columns.tolist() if a != cluster_col]
	mean_columns = [str(col) + '_mean' for col in columns]
	data2 = data1[columns].values - data1[mean_columns].values
	data2 = DataFrame(data2)
	data2.columns = columns
	data2[cluster_col] = clusters
	return data2

def get_wss_tss_ratio(data1, cluster_col = 'cluster', stages = None):
	composite = False
	if stages is not None:
		cluster_cols = data1.columns[Series(data1.columns).apply(lambda x: cluster_col in x)]
		data1[cluster_col] = data1[cluster_cols].apply(lambda x: ''.join(x.map(str)), axis=1)
		composite = True
		data1 = data1.drop(cluster_cols, axis = 1)
	data1_mean_diff = process_cluster_df(data1, cluster_col, composite)
	data1_total_wss = (data1_mean_diff.drop(cluster_col, axis=1)**2).sum().sum()
	data1_tss = (data1.drop(cluster_col, axis=1)**2).sum().sum()
	return data1_total_wss/data1_tss

def label_data(data, index, cluster_labels):
	if index is None:
		data['cluster'] = cluster_labels
	else:
		data['cluster_' + index] = cluster_labels
	return data

def identify_scale_normalize_pca(desc_df, continuous):
	scale = desc_df.index
	scale = scale[(desc_df["Outliers"] < 0.05) & (desc_df["vif"] < 4.0)]
	normalize = list(set(continuous) - set(scale))
	pca = desc_df.index
	pca = pca[(desc_df["vif"] >= 4.0)]
	non_pca = list(set(continuous) - set(pca.tolist()))
	transforms = {}
	transforms["scale"] = scale.tolist()
	transforms["normalize"] = normalize
	transforms["pca"] = pca.tolist()
	transforms["non_pca"] = non_pca
	return transforms

def cbind_matrix(mat1, mat2):
	lis = [mat1.reset_index(drop=True), mat2.reset_index(drop=True)]
	mat = concat(lis, axis = 1)
	return mat

def get_pca_all_vars(transforms, stages, stage, autoenc_cols):
	if autoenc_cols is None:
		pca_vars = set(transforms['pca'])
		pca_vars = pca_vars.intersection(stages[stage])
		non_pca_vars = set(stages[stage]) - pca_vars
		pca_vars = range(len(pca_vars))
		pca_vars = [stage + "_" + str(i) for i in pca_vars]
		all_vars = list(pca_vars) + list(non_pca_vars)
		return all_vars
	else:
		pca_vars = set(autoenc_cols)
		stage_vars = stages[stage]
		stage_vars = set([stage + "_" + str(i) for i in range(len(stage_vars))])
		all_vars = list(pca_vars.intersection(stage_vars))
		return all_vars

def floatify(x):
	try:
		return float(x)
	except:
		return NaN

def filter_data(data, filters):
	if "not_in" in filters.keys():
		not_in = filters["not_in"]
		for col in not_in.keys():
			try:
				data = data[data[col].apply(lambda x: x not in not_in[col])]
			except:
				continue
	if "in" in filters.keys():
		ins = filters["in"]
		for col in ins.keys():
			try:
				data = data[data[col].apply(lambda x: x in ins[col])]
			except:
				continue
	if "equals" in filters.keys():
		equals = filters["equals"]
		for col in equals.keys():
			try:
				data = data[data[col].apply(lambda x: x == equals[col])]
			except:
				continue
	if "not_equals" in filters.keys():
		not_equals = filters["not_equals"]
		for col in not_equals.keys():
			try:
				data = data[data[col].apply(lambda x: x!= not_equals[col])]
			except:
				continue
	if "not_nan" in filters.keys():
		not_nan = filters["not_nan"]
		for col in not_nan:
			try:
				data = data[data[col].apply(lambda x: not(isnan(floatify(x))))]
			except:
				continue
	return data
