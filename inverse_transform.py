from sklearn import decomposition
from pickle import load
from pandas import concat, DataFrame

def inverse_transform_pca_kpca(centroid, model):
    return DataFrame(model.inverse_transform(centroid))

def get_stage_model(stage, model_type):
    mdl = ""
    if model_type == "kpca":
        mdl = "k"
    if model_type != "autoencoder":
        model = load(open("datascience/" + mdl + "pca_model_" + stage + ".pkl", "rb"))
    else:
        from keras.models import load_model
        model = load_model("datascience/decoder_" + stage + ".hdf5")
    return model

def get_stage_names(stage, model_type):
    mdl = ""
    if model_type != "autoencoder":
        if model_type == "kpca":
            mdl = "k"
        vars = load(open("datascience/" + mdl + "pca_vars_" + stage + ".pkl", "rb")).tolist()
    else:
        vars = load(open("datascience/autoencoder_vars_" + stage + ".pkl", "rb")).tolist()
    return vars

def inverse_transform_autoencoder(centroid, decoder):
    out = DataFrame(decoder.predict(centroid))
    return out

def inverse_transform_stage(centroid, model_type, stage):
    model = get_stage_model(stage = stage, model_type = model_type)
    if model_type != "autoencoder":
        centroids = inverse_transform_pca_kpca(centroid = centroid, model = model)
    else:
        centroids = inverse_transform_autoencoder(centroid = centroid, decoder = model)
    
    names = get_stage_names(stage = stage, model_type = model_type)
    centroids.columns = names
    return centroids

def inverse_transform(kmeans_data, model_type, stages, max_vars = 10):
    inverse = {}
    for stage in stages.keys():
        try:
            stage_vars = []
            for i in range(max_vars):
                try:
                    var = kmeans_data[stage + "_" + str(i)]
                    stage_vars.append(stage + "_" + str(i))
                except:
                    break
            
            cluster_var = "cluster_" + stage
            kmeans_data[cluster_var] = kmeans_data[cluster_var].apply(str)
            if len(stage_vars) > 0:
                centroid = kmeans_data[stage_vars + [cluster_var]].groupby(cluster_var).mean().reset_index(drop = False)
                df = inverse_transform_stage(centroid = centroid, model_type = model_type, stage = stage)
            else:
                df = kmeans_data[stages[stage] + [cluster_var]].groupby(cluster_var).mean().reset_index(drop = False)
                # names = get_stage_names(stage = stage, model_type = model_type)
                # df.columns = names
            inverse[stage] = df
        
        except:
            pass
    
    return inverse
