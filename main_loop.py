from eda import run_clustering
from pandas import read_csv
from numpy import unique
from json import load, dump
from os import chdir

in_file = "data/Digital Revolution - Ransomware/gcs_misc.digital_revo_temp_table_10_May.csv"
in_cols = ["vertical_market_top"]

vals = read_csv(in_file, usecols = in_cols)
vals = unique(vals["vertical_market_top"])
vals = vals[vals.apply(type)!=float].tolist()

json_obj = load(open("datascience/filters.json", "r"))
modify = json_obj["equals"]
input_params = load(open("datascience/run_eda.json"))
run_cluster_analysis = input_params["run_cluster_analysis"]
# run_ba = input_params["run_ba"]
clustering_location = input_params["cluster_location"]
filters_file = input_params["filters"]
transforms_file = input_params["transforms"]
stages_file = input_params["stages"]
cluster_combine_file = input_params["cluster_combine"]
run_kmeans_model = input_params["run_kmeans_model"]
run_hierarchical_model = input_params["run_hierarchical_model"]
run_pca_model = input_params["run_pca_model"]
run_kpca_model = input_params["run_kpca_model"]
run_autoencoder_model = input_params["run_autoencoder_model"]

for val in vals:
    modify["vertical_market_top"] = val
    json_obj["equals"] = modify
    dump(json_obj, open("datascience/filters.json", "w"))    
    # basket_config_file = input_params["basket_config"]
    # basket_location = input_params["basket_location"]
    
    try:
        if run_cluster_analysis:
	        run_clustering(clustering_location, filters_file, transforms_file, stages_file, cluster_combine_file, run_kmeans_model, run_hierarchical_model, run_pca_model, run_kpca_model, run_autoencoder_model, outfile_prefix = val)
    except:
        continue

