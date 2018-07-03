# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 15:00:42 2018

@author: nitin.s, naveen.nathan
"""
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet, fcluster
from scipy.spatial.distance import pdist
from util import label_data, get_pca_all_vars
from pandas import concat
from pickle import dump
from numpy import unique

def run_kmeans(data, stages = None, index = None, transforms = None, autoenc_cols = None, outfile_prefix = ""):
    if stages is None:
        n_cluster = optimal_k_silhouette(data, [i+2 for i in range(9)])
        kmeans = KMeans(init = 'k-means++', n_clusters =  n_cluster, n_init = 10)
        if index is None:
            dump(kmeans, open("datascience/" + outfile_prefix + "kmeans_model.pkl", "wb"))
        else:
            dump(kmeans, open("datascience/" + outfile_prefix + "kmeans_model_" + index + ".pkl", "wb"))
        cluster_labels = kmeans.fit_predict(data)
        data = label_data(data, index, cluster_labels)
    else:
        data1 = []
        for stage in stages.keys():
            if transforms is None:
                data1 = data1 + [run_kmeans(data[stages[stage]], index = stage, outfile_prefix = outfile_prefix)]
            else:
                all_vars = get_pca_all_vars(transforms, stages, stage, autoenc_cols)
                data1 = data1 + [run_kmeans(data[all_vars], index = stage, outfile_prefix = outfile_prefix)]
        data = concat(data1, axis = 1)
    return data


def run_hierarchical_clustering(data, stages = None, index = None, transforms = None, autoenc_cols = None, outfile_prefix = ""):
    if stages is None:
        n_clusters = optimal_k_silhouette(data, [i+2 for i in range(9)])
        h_clust = linkage(data, 'ward')
        c, coph_dists = cophenet(h_clust, pdist(data))
        if index is None:
            dump(h_clust, open("datascience/" + outfile_prefix + "hclust_model.pkl", "wb"))
        else:
            dump(h_clust, open("datascience/" + outfile_prefix + "hclust_model_" + index + ".pkl", "wb"))
        cluster_labels = fcluster(h_clust, n_clusters, criterion = 'maxclust')
        data = label_data(data, index, cluster_labels)
    else:
        data1 = []
        for stage in stages.keys():
            if transforms is None:
                data1 = data1 + [run_hierarchical_clustering(data[stages[stage]], index = stage, autoenc_cols = autoenc_cols, outfile_prefix = outfile_prefix)]
            else:
                all_vars = get_pca_all_vars(transforms, stages, stage, autoenc_cols)
                data1 = data1 + [run_hierarchical_clustering(data[all_vars], index = stage, autoenc_cols = autoenc_cols, outfile_prefix = outfile_prefix)]
        data = concat(data1, axis = 1)
    return data


def optimal_k_silhouette(data, range_n_clusters):
    from sklearn.metrics import silhouette_score
    range_n_clusters = range_n_clusters
    silhouette_max = -1
    optimum_cluster = range_n_clusters[1]
    for n_clusters in range_n_clusters:
        kmeans = KMeans(n_clusters = n_clusters, random_state = 10, n_init = 10)
        cluster_labels = kmeans.fit_predict(data)
        silhouette_avg = silhouette_score(data, cluster_labels)
        if silhouette_max < silhouette_avg :
            silhouette_max = silhouette_avg
            optimum_cluster = n_clusters
    return optimum_cluster


def optimal_k_elbow(data, range_n_clusters):
    from pandas import DataFrame
    optimum_cluster = 2
    range_n_clusters = range_n_clusters
    cluster_errors = []
    for num_clusters in range_n_clusters:
        clusters = KMeans( num_clusters )
        clusters.fit(data)
        cluster_errors.append( clusters.inertia_ )
    clusters_df = DataFrame( { "num_clusters":range_n_clusters, "cluster_errors": cluster_errors } )
    secondDerivative = 0
    for i in range(1,clusters_df.shape[0]):
        temp_2nd_derivative = clusters_df.ix[i+1, 'cluster_errors'] + clusters_df.ix[i-1, 'cluster_errors'] - 2*clusters_df.ix[i, 'cluster_errors']
        if abs(temp_2nd_derivative) > secondDerivative : 
            secondDerivative = abs(temp_2nd_derivative)
            optimum_cluster = clusters_df.ix[i, 'num_clusters']
    return optimum_cluster


def optimal_k_gap(data, range_n_clusters):
    from gap import gap_statistic, find_optimal_k
    gaps, s_k, K = gap_statistic(data, refs = None, B = 10, K = range_n_clusters, N_init = 10)
    bestKValue = find_optimal_k(gaps, s_k, K)
    return bestKValue


def churn_clusters(data, dependent = "churn"):
    from CHAID import Tree
    indep_cols = list(set(data.columns.tolist()) - set(dependent))
    tree = Tree.from_pandas_df(data, dict(zip(indep_cols,
        ['nominal'] * len(indep_cols))), dependent, min_child_node_size=5)
    print(tree.print_tree())


def churn_clusters(data, dependent = "churn"):
    from sklearn.tree import DecisionTreeClassifier
    dt = DecisionTreeClassifier(criterion = "gini", random_state = 1, max_depth = 4, min_samples_leaf = 4)
    dt.fit(data.drop([dependent], axis = 1), data[dependent])
    preds = dt.predict_proba(data.drop([dependent], axis = 1))
    data["preds"] = preds
    return data

