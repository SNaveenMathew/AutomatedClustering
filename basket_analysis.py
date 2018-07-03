from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from pandas import Series, DataFrame#, read_excel
from re import findall, sub
from itertools import chain

#df = read_excel('../DR- template_real_data.xlsx')
#df.head()
# uniq_cust = set(df['primary_key'].tolist())

def preprocess_df(df, primary_key, duplicate_check, offer_cols, patt, c_type):
	df[primary_key] = df[primary_key].apply(lambda x: str(x).strip())
	pk = df[primary_key]
	if c_type != "binary":
		if duplicate_check is not None:
			reqd_cols_list = [primary_key, duplicate_check]
			other_reqd = Series(df.columns).apply(lambda x: findall(string = x,
				pattern = patt))
			other_reqd = other_reqd[other_reqd.apply(len)>0]
			reqd_cols_list = reqd_cols_list + list(chain(*other_reqd.tolist()))
			df = df[reqd_cols_list]
			df[primary_key] = df[primary_key].apply(str) + df[duplicate_check].apply(str)
			df = df.drop([duplicate_check], axis = 1)
		
		# df = df.drop([primary_key], axis = 1)
		offer_cols = [x for x in list(df.columns) if len(findall(string = x,
		pattern = patt)) > 0]
		# offer_cols = offer_cols[2:-1]
		df = df[offer_cols].apply(lambda x: x.apply(type) == str)
		df[primary_key] = pk
	
	else:
		offer_cols = [x for x in df.columns.tolist() if x != primary_key]
	
	df = df.groupby([primary_key]).sum()
	df = df[offer_cols]
	df = df.apply(lambda x: (x>0).apply(int), axis=1)
	df.columns = [sub(string = col, pattern = sub(pattern = "^.{2}", repl = "", string = patt),
	repl = "") for col in list(df.columns)]
	df = df[df.sum(axis = 1)!=0]
	return df


def perform_basket_analysis(df, primary_key = 'Customer_Name',
 duplicate_check = 'Derived_Party_ID', offer_cols = None, colpattern = ".*_status_name$",
 c_type = "binary", min_support = 0.001, min_threshold = 0.01):
	df[primary_key] = df[primary_key].apply(str)
	df = preprocess_df(df, primary_key, duplicate_check, offer_cols,
	patt = colpattern, c_type = c_type)
	from numpy import sum
	df['Total'] = df.apply(sum, axis = 1)
	for col in df.columns.tolist():
		col_df = df.groupby([col]).mean()
		col_df.to_csv("datascience/" + col + "_summary.csv")
	
	df = df.drop(['Total'], axis = 1)
	frequent_itemsets = apriori(df, min_support = min_support, use_colnames = True)
	frequent_itemsets.to_csv("../frequent_itemset_s=" + str(min_support) + ".csv")
	rules = association_rules(frequent_itemsets, metric = "confidence",
		min_threshold = min_threshold)
	rules.to_csv("../association_rules_s=" + str(min_support) + "_c=" + str(min_threshold) + "_py.csv")
	return rules
