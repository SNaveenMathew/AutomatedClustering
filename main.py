from eda import run_clustering, run_basket_analysis
from json import load

input_params = load(open("datascience/run_eda.json"))
run_cluster_analysis = input_params["run_cluster_analysis"]
run_ba = input_params["run_ba"]
clustering_location = input_params["cluster_location"]
filters_file = input_params["filters"]
transforms_file = input_params["transforms"]
stages_file = input_params["stages"]
cluster_combine_file = input_params["cluster_combine"]
basket_config_file = input_params["basket_config"]
basket_location = input_params["basket_location"]
run_kmeans_model = input_params["run_kmeans_model"]
run_hierarchical_model = input_params["run_hierarchical_model"]
run_pca_model = input_params["run_pca_model"]
run_kpca_model = input_params["run_kpca_model"]
run_autoencoder_model = input_params["run_autoencoder_model"]

if run_cluster_analysis:
	run_clustering(clustering_location, filters_file, transforms_file, stages_file, cluster_combine_file, run_kmeans_model, run_hierarchical_model, run_pca_model, run_kpca_model, run_autoencoder_model)

if run_ba:
	run_basket_analysis(basket_location, basket_config_file)
