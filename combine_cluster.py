from json import load
from re import sub
from pandas import DataFrame

def combine_clusters(data, centroids, stages, cluster_combine, cluster_col = "cluster"):
    ratio = cluster_combine['ratio']
    for stage in stages.keys():
        if stage in centroids.keys():
            stage_vars = stages[stage]
            for key in ratio.keys():
                if key in stage_vars:
                    col = cluster_col + "_" + stage
                    cols = [key, col]
                    df = DataFrame(centroids[stage][cols])
                    df['dummy'] = 1
                    df = df.merge(df, on = ['dummy'], how = "outer")
                    df1 = df[df[col + "_x"] != df[col + "_y"]]
                    df = df1[df1[key + "_x"] > df1[key + "_y"]]
                    df['ratio'] = df[key + "_x"] / df[key + "_y"]
                    df1['ratio'] = df1[key + "_x"] / df1[key + "_y"]
                    cols = [col + "_x", 'ratio']
                    df = df[cols].groupby([col + "_x"]).min().reset_index(drop = False)
                    df['combine'] = df['ratio'] < ratio[key]
                    df1 = df1.merge(df, on = [col + "_x", "ratio"], how = "outer")
                    df1 = df1.drop(["dummy", key + "_y", key + "_x"], axis = 1)
                    df1.columns = [sub(pattern = "_x$", repl = "", string = x) for x in df1.columns.tolist()]
                    df1[col] = df1[col].apply(str)
                    data[col] = data[col].apply(str)
                    data = data.merge(df1, on = [col], how = "outer")
                    condition = data['combine'].apply(type) == float
                    data['combine'][condition] = False
                    data[col][~condition] = data[col + "_y"][~condition]
                    data = data.drop([col + "_y", "combine"], axis = 1)
                
            
        
    return data
