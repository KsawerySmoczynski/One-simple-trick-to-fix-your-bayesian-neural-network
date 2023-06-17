import pandas as pd
import numpy as np

model_dic = {
	'PyroMLEClassify': '1FC',
	'PyroDeepMLEClassify': '3FC',
	'PyroConvClassify': 'Conv'
}

df = pd.read_csv('metric_stats.csv')
# print(df.columns)

df.columns = ['index', 'model', 'data', 'slope', 'Acc_mean', 'Acc_std', 'ECE_mean', 'ECE_std']
df['var'] = 'auto'
df['NLL_mean'] = np.inf
df['NLL_std'] = np.nan

df = df[df['slope'].str.contains('negative')]

df['slope'] = df['slope'].str[29:-1]

for col in ['Acc_mean', 'Acc_std', 'ECE_mean', 'ECE_std']:
	df[col] = round(df[col], 3)

df['model'] = df['model'].replace(model_dic)

df.to_csv('auto_stats.csv', index=False)