import pandas as pd
from matplotlib import pyplot as plt
import os

def get_metrics(var):
	metrics = ['ECE', 'Acc']
	if var == 'manual':
		metrics.append('NLL')

	return metrics

for var in ['auto', 'manual']:
	df = pd.read_csv(var+'_stats.csv')
	os.makedirs(var+'_plots', exist_ok=True)

	for data in df['data'].unique():
		for model in df['model'].unique():
			tdf = df[(df['data'] == data) & (df['model'] == model)]
			for metric in get_metrics(var):

				fig, ax = plt.subplots()
				fig.set_size_inches(7, 2)
				plt.xlabel('negative slope')
				plt.ylabel(metric)
				ax.errorbar(tdf['slope'], tdf[metric+'_mean'], tdf[metric+'_std'], marker='o', linestyle='none')
				plt.savefig(f"{var}_plots/{data}-{model}-{metric}.png", bbox_inches = "tight", dpi=1000)
				plt.close(fig)