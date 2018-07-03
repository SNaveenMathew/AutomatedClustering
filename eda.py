from pandas import read_csv, DataFrame#, read_excel#, Series
from transform import run_autoencoder, run_tsne
from descriptive import descriptives_continuous, corr_mat#, descriptives_categorical
from inference import run_all_f_test
from plot import plot_raw_data#, plot_processed_data
from transform import transform_continuous, run_pca#, transform_categorical, run_kernel_pca
from model import run_kmeans, run_hierarchical_clustering
from util import get_wss_tss_ratio, identify_scale_normalize_pca, cbind_matrix, filter_data
from numpy import unique
from json import load
from basket_analysis import perform_basket_analysis
from os.path import isfile
from inverse_transform import inverse_transform
from pandas.io.parsers import ExcelFile
from os import chdir
from re import sub
from numpy import NaN
from combine_cluster import combine_clusters

chdir("../")

def run_clustering(clustering_location, filters_file, transforms_file, stages_file, cluster_combine_file, run_kmeans_model, run_hierarchical_model, run_pca_model, run_kpca_model, run_autoencoder_model, outfile_prefix = ""):
	remove_cols = ["cluster_stage_1", "cluster_stage_2", "cluster_stage_3", "cluster_stage_4", "Composite_Cluster", "comp_cluster_wo_usertype", "profile"]
	try:
		data = read_csv(clustering_location)
	except:
		data = read_csv(clustering_location, encoding = "latin1")
	
	for col in remove_cols:
		try:
			data = data.drop([col], axis = 1)
		except:
			continue
	
	if isfile(filters_file):
		filters = load(open(filters_file))
		data = filter_data(data, filters)
	
	transforms = load(open(transforms_file))
	stages = load(open(stages_file))
	transforms["stages"] = stages
	if "continuous" not in transforms.keys():
		transforms["continuous"] = []
		for stage in stages.keys():
			transforms["continuous"] = transforms["continuous"] + stages[stage]

	print(transforms["continuous"])
	print(data.columns)
	data[transforms["continuous"]] = data[transforms["continuous"]].apply(lambda x: x.apply(float), axis = 0)
	data[transforms["continuous"]] = data[transforms["continuous"]].fillna(0, axis = 1)
	desc_df = descriptives_continuous(data, transforms["continuous"], stages)
	corr_matrix = corr_mat(data, transforms["continuous"])
	transforms_new = identify_scale_normalize_pca(desc_df, transforms["continuous"])
	# plot_raw_data(data, corr_matrix, transforms)
	if "categorical" in transforms.keys():
		if len(transforms["categorical"]) > 0:
			run_all_f_test(data, transforms)

	data = transform_continuous(data, transforms_new)
	if run_pca_model:
		pca_mat = run_pca(data = data, stages = transforms["stages"], pca_vars = transforms_new['pca'])
		pca_mat = DataFrame(pca_mat)
		pca_mat = cbind_matrix(pca_mat, data[transforms_new["non_pca"]])
	
	if run_kpca_model:
		kpca_mat = run_pca(data = data, stages = transforms["stages"], pca_vars = transforms_new['pca'],
		pca_type = "kernel", kernel = "rbf")
		kpca_mat = DataFrame(kpca_mat)
		kpca_mat = cbind_matrix(kpca_mat, data[transforms_new["non_pca"]])

	if run_autoencoder_model:
		autoenc_mat = run_autoencoder(data = data, stages = transforms["stages"])
	
	kmeans_data1_ratio = 1
	hierarchical_data1_ratio = 1
	kmeans_pca_ratio = 1
	kmeans_kpca_ratio = 1
	kmeans_autoencoder_ratio = 1
	hierarchical_pca_ratio = 1
	hierarchical_kpca_ratio = 1
	hierarchical_autoencoder_ratio = 1

	if run_kmeans_model:
		kmeans_data1 = run_kmeans(data[transforms["continuous"]], stages = stages, outfile_prefix = outfile_prefix)
		if run_pca_model:
			kmeans_data2 = run_kmeans(data = pca_mat, stages = stages, transforms = transforms_new, outfile_prefix = outfile_prefix)
			# Centroids data frame may have different number of rows for different stages
			# Therefore, few rows may have NaN
			kmeans_data2_centroids = inverse_transform(kmeans_data = kmeans_data2, stages = stages, model_type = "pca", max_vars = len(transforms["continuous"]))
			kmeans_pca_ratio = get_wss_tss_ratio(kmeans_data2, 'cluster', stages) # 0.33250593935370332
		if run_kpca_model:
			kmeans_data3 = run_kmeans(data = kpca_mat, stages = stages, transforms = transforms_new, outfile_prefix = outfile_prefix)
			# Centroids data frame may have different number of rows for different stages
			# Therefore, few rows may have NaN
			kmeans_data3_centroids = inverse_transform(kmeans_data = kmeans_data3, stages = stages, model_type = "kpca", max_vars = len(transforms["continuous"]))
			kmeans_kpca_ratio = get_wss_tss_ratio(kmeans_data3, 'cluster', stages) # 0.32357203754255803
		if run_autoencoder_model:
			kmeans_data4 = run_kmeans(data = autoenc_mat, stages = stages, transforms = transforms_new, autoenc_cols = autoenc_mat.columns.tolist(), outfile_prefix = outfile_prefix)
			# Centroids data frame may have different number of rows for different stages
			# Therefore, few rows may have NaN
			kmeans_data4_centroids = inverse_transform(kmeans_data = kmeans_data4, stages = stages, model_type = "autoencoder", max_vars = len(transforms["continuous"]))
			kmeans_autoencoder_ratio = get_wss_tss_ratio(kmeans_data4, 'cluster', stages) # 0.21651274
	
	if run_hierarchical_model:
		hierarchical_data1 = run_hierarchical_clustering(data = data, stages = stages, outfile_prefix = outfile_prefix)
		if run_pca_model:
			hierarchical_data2 = run_hierarchical_clustering(data = pca_mat, stages = stages, transforms = transforms_new, outfile_prefix = outfile_prefix)
			hierarchical_data2_centroids = inverse_transform(kmeans_data = hierarchical_data2, stages = stages, model_type = "kpca", max_vars = len(transforms["continuous"]))
			hierarchical_pca_ratio = get_wss_tss_ratio(hierarchical_data2, 'cluster', stages) # 0.33250593935370332
		if run_kpca_model:
			hierarchical_data3 = run_hierarchical_clustering(data = kpca_mat, stages = stages, transforms = transforms_new, outfile_prefix = outfile_prefix)
			hierarchical_data3_centroids = inverse_transform(kmeans_data = hierarchical_data3, stages = stages, model_type = "kpca", max_vars = len(transforms["continuous"]))
			hierarchical_kpca_ratio = get_wss_tss_ratio(hierarchical_data3, 'cluster', stages) # 0.32309567761955454 -> Best
		if run_autoencoder_model:
			hierarchical_data4 = run_hierarchical_clustering(data = autoenc_mat, stages = stages, transforms = transforms_new, autoenc_cols = autoenc_mat.columns.tolist(), outfile_prefix = outfile_prefix)
			hierarchical_data4_centroids = inverse_transform(kmeans_data = hierarchical_data4, stages = stages, model_type = "autoencoder", max_vars = len(transforms["continuous"]))
			hierarchical_autoencoder_ratio = get_wss_tss_ratio(hierarchical_data4, 'cluster', stages) # 0.21349777

	desc_df.to_csv("datascience/" + outfile_prefix + "desc_df.csv")
	corr_matrix.to_csv("datascience/" + outfile_prefix + "corr_matrix.csv")
	try:
		kmeans_data1_ratio = get_wss_tss_ratio(kmeans_data1, 'cluster', stages) # 0.32776762109162327
	except:
		pass
	try:
		hierarchical_data1_ratio = get_wss_tss_ratio(hierarchical_data1, 'cluster', stages) # 0.32776762109162327
	except:
		pass
	
	best = min(kmeans_data1_ratio, kmeans_pca_ratio, kmeans_kpca_ratio, kmeans_autoencoder_ratio,
		hierarchical_data1_ratio, hierarchical_pca_ratio, hierarchical_kpca_ratio, hierarchical_autoencoder_ratio)
	print(best)
	
	try:
		if kmeans_data1_ratio == best:
			kmeans_data1.to_csv("datascience/" + outfile_prefix + "kmeans_data_mat.csv")
	except:
		pass
	try:
		if kmeans_pca_ratio == best:
			kmeans_data2.to_csv("datascience/" + outfile_prefix + "kmeans_pca_mat.csv")
			kmeans_data2_centroids.to_csv("datascience/" + outfile_prefix + "kmeans_pca_original_centroid.csv")
	except:
		pass
	try:
		if kmeans_kpca_ratio == best:
			kmeans_data3.to_csv("datascience/" + outfile_prefix + "kmeans_kpca_mat.csv")
			kmeans_data3_centroids.to_csv("datascience/" + outfile_prefix + "kmeans_kpca_original_centroid.csv")
	except:
		pass
	try:
		if kmeans_autoencoder_ratio == best:
			kmeans_data4.to_csv("datascience/" + outfile_prefix + "kmeans_autoencoder_mat.csv")
			kmeans_data4_centroids.to_csv("datascience/" + outfile_prefix + "kmeans_autoencoder_original_centroid.csv")
	except:
		pass
	try:
		if hierarchical_data1_ratio == best:
			hierarchical_data1.to_csv("datascience/" + outfile_prefix + "hierarchical_data_mat.csv")
	except:
		pass
	try:
		if hierarchical_pca_ratio == best:
			hierarchical_data2.to_csv("datascience/" + outfile_prefix + "hierarchical_pca_mat.csv")
			hierarchical_data2_centroids.to_csv("datascience/" + outfile_prefix + "hierarchical_pca_original_centroid.csv")
	except:
		pass
	try:
		if hierarchical_kpca_ratio == best:
			hierarchical_data3.to_csv("datascience/" + outfile_prefix + "hierarchical_kpca_mat.csv")
			hierarchical_data3_centroids.to_csv("datascience/" + outfile_prefix + "hierarchical_kpca_original_centroid.csv")
	except:
		pass
	try:
		if hierarchical_autoencoder_ratio == best:
			hierarchical_data4.to_csv("datascience/" + outfile_prefix + "hierarchical_autoencoder_mat.csv")
			hierarchical_data4_centroids.to_csv("datascience/" + outfile_prefix + "hierarchical_autoencoder_original_centroid.csv")
	except:
		pass
	
	if isfile(cluster_combine_file):
		cluster_combine = load(open(cluster_combine_file))
		kmeans_data2 = combine_clusters(kmeans_data2, kmeans_data2_centroids, stages, cluster_combine, "cluster")



# Market basket based clusters
#basket_df = ExcelFile("../data/DR- template_real_data.xlsx").parse('Data')
def run_basket_analysis(basket_location, basket_config_file):
	from sklearn.preprocessing import MultiLabelBinarizer
	try:
		basket_df = read_csv(basket_location)
	except:
		basket_df = read_csv(basket_location, encoding = "latin1")

	basket_df.columns = [sub(pattern = "digital_revo_final_calculated_fields\.", repl = "",
	string = x) for x in basket_df.columns.tolist()]
	basket_df.columns = [sub(pattern = "_bought$", repl = "s", string = x) for x in basket_df.columns.tolist()]
	# primary_key = 'party_id'
	basket_config = load(open(basket_config_file))
	primary_key = basket_config['primary_key']
	products_col = basket_config['products_col']
	colpattern = basket_config['colpattern']
	duplicate_check = basket_config['duplicate_check']
	c_type = basket_config['c_type']
	min_support = basket_config['min_support']
	min_threshold = basket_config['min_threshold']
	basket_df.columns = [sub(pattern = "^party_id$", repl = primary_key, string = x) for x in basket_df.columns.tolist()]
	pk = basket_df[primary_key]
	# products_col = 'license_features'
	products_col = sub(pattern = "_bought$", repl = "s", string = products_col)
	basket_df[products_col] = basket_df[products_col].apply(lambda x: sub(pattern="[\[\]\"\']",
		repl="", string = str(x)).split(","))
	basket_df[products_col] = basket_df[products_col].apply(lambda x: [sub(pattern = "^ {1,}| {1,}$", repl = "", string = offer) for offer in x])
	condition = basket_df[products_col].apply(lambda x: x!=['nan'])
	basket_df[products_col][condition] = basket_df[products_col][condition].apply(lambda x: [y + "_offer" for y in x])
	mlb = MultiLabelBinarizer()
	df = mlb.fit_transform(basket_df[products_col])
	df = DataFrame(df, columns = mlb.classes_, index = basket_df.index)
	df.columns = [sub(pattern = "^ {1,}| {1,}$", repl = "", string = col) for col in df.columns.tolist()]
	if 'nan' in df.columns:
		df = df.drop(['nan'], axis = 1)

	if '_offer' in df.columns:
		df = df.drop(['_offer'], axis = 1)

	df[primary_key] = pk.apply(str)
	basket_df = df
	del df
	#duplicate_check = 'Derived_Party_ID'

	# Primary key: Unique identifier for a customer
	# Duplicate check: To check whether there are 2 different customers with same unique identifier
	# or whether 2 unique identifiers refer to the same customer
	# If '_status_name' field is not empty, that offer has been subscribed
	rules = perform_basket_analysis(basket_df, primary_key = primary_key,
		duplicate_check = duplicate_check, colpattern = colpattern, c_type = c_type,
		min_support = min_support, min_threshold = min_threshold)
	return rules


# Churn based clusters
# churn = churn_clusters(data, "churn")
# churn_clusters_df = DataFrame(Series(unique(churn["preds"])))
# churn_clusters_df["cluster"] = range(churn_clusters_df.shape[0])