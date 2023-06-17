import pandas as pd
from matplotlib import pyplot as plt
import os

df = pd.read_csv('stats.csv')

df['PICP_mean'] = df['PICP_mean'] / 100
df['PICP_std'] = df['PICP_std'] / 100

for var in ['auto', 'manual']:
	os.makedirs(var+'_plots', exist_ok=True)
	tdf = df[df['var'] == var]
	for metric in ['RMSE', 'PICP', 'MPIW']:

		fig, ax = plt.subplots()
		fig.set_size_inches(7, 2)
		plt.xlabel('negative slope')
		plt.ylabel(metric)

		if metric == 'PICP':
			plt.axhline(y = 0.95, color = 'r', linestyle = '-')

		ax.errorbar(tdf['slope'], tdf[metric+'_mean'], tdf[metric+'_std'], marker='o', linestyle='none')
		plt.savefig(f"{var}_plots/{metric}.png", bbox_inches = "tight", dpi=1000)
		plt.close(fig)