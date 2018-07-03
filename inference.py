import statsmodels.api as sm
from statsmodels.formula.api import ols
from pandas import DataFrame
from scipy.stats import f_oneway

def run_all_f_test(data, transforms):
    continuous = transforms['continuous']
    categorical = transforms['categorical']
    for cat in categorical:
        cat_col = data[cat]
        data[continuous].apply(lambda x: run_f_test(x, cat_col), axis = 0)

def run_f_test(series, group):
    col_name = series.name
    cat_name = group.name
    df = DataFrame(series)
    df[cat_name] = group
    mod = ols(col_name + ' ~ ' + cat_name, data = df).fit()
    aov_table = sm.stats.anova_lm(mod, typ = 2)
    aov_table.to_csv("datascience/" + col_name + "_" + cat_name + ".csv")

def run_group_f_test(series, group):
    col_name = series.name
    cat_name = group.name
    df = DataFrame(series)
    df[cat_name] = group
    dic = {}
    for i in range(len(series)):
        try:
            dic[cat_name[i]] = dic[cat_name[i]] + [series[i]]
        except:
            dic[cat_name[i]] = [series[i]]
        
    
    samples = [x[col_name] for x in df.groupby(cat_name)]
    f_oneway(*samples)