from ittk import entropy, information_variation, calc_MI
from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrices
from transform import transform_continuous
from pandas import Series

def descriptives_continuous(data, continuous, stages = None):
	data = data[continuous]
	desc_df = data.describe()
	indices = desc_df.index
	df1 = data.skew()
	desc_df = desc_df.append(df1, ignore_index = True)
	df2 = data.kurt()
	indices = list(indices) + ['skewness', 'kurtosis', 'vif']
	desc_df = desc_df.append(df2, ignore_index = True)
	if(stages is None):
		vifs = vif(data, continuous)
	else:
		vifs = []
		for stage in stages:
			stage_vars = stages[stage]
			if len(stage_vars) > 1:
				vifs = vifs + vif(data[stage_vars], stage_vars)
			else:
				vifs = vifs + [1]
	vifs = Series(vifs)
	vifs.index = desc_df.columns
	desc_df = desc_df.append(vifs, ignore_index = True)
	desc_df.index = indices
	desc_df = desc_df.T
	desc_df["CoV"] = desc_df["mean"]/desc_df["std"]
	desc_df["IQR"] = desc_df["75%"] - desc_df["25%"]
	desc_df["Range"] = desc_df["max"] - desc_df["min"]
	transforms = {}
	transforms["scale"] = continuous
	data = transform_continuous(data, transforms)
	desc_df["Outliers"] = count_outliers_scaled(data)
	return desc_df

def descriptives_categorical(data, categorical):
	data = data[categorical]
	mi = calc_MI()

def corr_mat(data, continuous):
	data = data[continuous]
	corr = data.corr()
	corr.index = continuous
	corr.columns = continuous
	return corr

def count_outliers(series):
	mean = series.mean()
	sd = series.std()
	lower = mean - 1.96*sd
	upper = mean + 1.96*sd
	out = series < lower
	out = out | series > upper
	return out.mean()

def vif(data, continuous):
	y, X = dmatrices("1~"+"+".join(continuous), data)
	X = X[:, 1:]
	return [variance_inflation_factor(X, i) for i in range(X.shape[1])]

def count_outliers_scaled(data):
	total = (data > 1.96).mean() + (data < -1.96).mean()
	return total
