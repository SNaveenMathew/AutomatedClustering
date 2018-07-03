from sklearn.preprocessing import scale, normalize, LabelEncoder, OneHotEncoder
from sklearn import decomposition
from numpy import cumsum, where
from pandas import concat, DataFrame
from pickle import dump
from math import ceil

def transform_continuous(data, transforms):
	try:
		if(len(transforms["scale"]) > 0):
			data[transforms["scale"]] = scale(data[transforms["scale"]])
	except:
		pass
	try:
		if(len(transforms["normalize"]) > 0):
			data[transforms["normalize"]] = normalize(data[transforms["normalize"]])
	except:
		pass
	return data


def transform_categorical(data, transforms):
	le = LabelEncoder()
	data2 = data[transforms['categorical']].apply(le.fit_transform)
	enc = OneHotEncoder()
	enc.fit(data2)
	onehotlabels = enc.transform(data2).toarray()
	return onehotlabels


def run_pca(data, threshold = None, stages = None, index = None, pca_vars = None, pca_type = "linear", kernel = "rbf", outfile_prefix = ""):
	if(pca_type == "linear"):
		pca = decomposition.PCA()
		char = ''
	else:
		pca = decomposition.KernelPCA(kernel = kernel)
		char = 'k'
	if stages is None:
		if pca_vars is not None:
			data = data[pca_vars]
		pca.fit(data)
		data2 = DataFrame(pca.transform(data))
		if threshold is not None:
			cs = cumsum(pca.explained_variance_ratio_) > threshold
			stop = where(cs)[0][0]
			data2 = data2[:, 0:(stop+1)]
		if index is None:
			dump(pca, open("datascience/" + outfile_prefix + "pca_model.pkl", 'wb'))
			dump(pca_vars, open("datascience/" + outfile_prefix + "pca_vars.pkl", "wb"))
		else:
			data2.columns = [index + "_" + str(col) for col in data2.columns]
			dump(pca, open("datascience/" + char + "pca_model_" +  index + ".pkl", "wb"))
			dump(data.columns, open("datascience/" + char + "pca_vars_" + index + ".pkl", "wb"))
	else:
		data2 = []
		for stage in stages:
			variables = stages[stage]
			pca_vars_new = list(set(pca_vars).intersection(set(variables)))
			if pca_vars_new is not None:
				if len(pca_vars_new) > 0:
					data2 = data2 + [run_pca(data[pca_vars_new], threshold = threshold, index = stage, pca_type = pca_type, kernel = kernel, outfile_prefix = outfile_prefix)]
		data2 = concat(data2, axis = 1)
	return data2


def run_tsne(data, n_components = 2):
	from sklearn.manifold import TSNE
	tsne = TSNE(n_components=2)
	X_tsne = tsne.fit_transform(data)
	return DataFrame(X_tsne)


def run_autoencoder(data, n_bottleneck = 2, stages = None, index = None, outfile_prefix = ""):
	from keras.layers import Input, Dense
	from keras.models import Model
	if stages is None:
		ncol = data.shape[1]
		input_layer = Input(shape = (ncol, ))
		encoded = Dense(n_bottleneck, activation = 'relu')(input_layer)
		decoded = Dense(ncol, activation = 'sigmoid')(encoded)
		autoencoder = Model(input_layer, decoded)
		encoder = Model(input_layer, encoded)
		encoded_input = Input(shape = (n_bottleneck, ))
		decoder_layer = autoencoder.layers[-1]
		decoder = Model(encoded_input, decoder_layer(encoded_input))
		autoencoder.compile(optimizer = 'adadelta', loss = 'binary_crossentropy')
		autoencoder.fit(data, data, epochs = 50, batch_size = 256, shuffle = True)
		if index is None:
			autoencoder.save("datascience/" + outfile_prefix + "autoencoder_model.hdf5")
			decoder.save("datascience/" + outfile_prefix + "decoder.hdf5")
			dump(data.columns, open("datascience/" + outfile_prefix + "autoencoder_vars.pkl", "wb"))
		else:
			autoencoder.save("datascience/" + outfile_prefix + "autoencoder_model_" + index + ".hdf5")
			decoder.save("datascience/" + outfile_prefix + "decoder_" + index + ".hdf5")
			dump(data.columns, open("datascience/" + outfile_prefix + "autoencoder_vars_" + index + ".pkl", "wb"))
		encoded_data = DataFrame(encoder.predict(data))
		encoded_data.columns = [index + "_" + str(col) for col in encoded_data.columns.tolist()]
	else:
		encoded_data = []
		for stage in stages.keys():
			stage_vars = stages[stage]
			data1 = data[stage_vars]
			n_bottleneck = ceil(len(stage_vars)/2)
			e_data = run_autoencoder(data = data1, n_bottleneck = n_bottleneck, index = stage, outfile_prefix = outfile_prefix)
			encoded_data.append(e_data)
		encoded_data = concat(encoded_data, axis = 1, ignore_index = False)
	return encoded_data

# def run_kernel_pca(data, kernel = 'rbf', threshold = None, num_dim = None):
# 	kpca = decomposition.KernelPCA(kernel = kernel)
# 	kpca.fit(data)
# 	data2 = kpca.transform(data)
# 	if num_dim is not None:
# 		data2 = data2[:, 0:num_dim]
# 	elif threshold is not None:
# 		explained_variance = data2.var()
# 		explained_variance_ratio_ = explained_variance/explained_variance.sum()
# 		cum_var_explained = cumsum(explained_variance_ratio_)
# 		cs = cum_var_explained > threshold
# 		stop = where(cs)[0][0]
# 		data2 = data2[:, 0:(stop+1)]
# 	dump(kpca, open("datascience/kpca_model.pkl", 'wb'))
# 	return data2

