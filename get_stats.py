import pandas as pd

df = pd.read_csv('results.csv')
stats = []

for var in df['var'].unique():
	for slope in df['slope'].unique():
		tdf = df[(df['var'] == var) &  (df['slope'] == slope)]
		
		mean_RMSE = round(tdf['RMSE'].mean(), 3)
		std_RMSE = round(tdf['RMSE'].std(), 3)

		mean_PICP = round(tdf['PICP'].mean(), 3)
		std_PICP = round(tdf['PICP'].std(), 3)

		mean_MPIW = round(tdf['MPIW'].mean(), 3)
		std_MPIW = round(tdf['MPIW'].std(), 3)

		stat = [var, slope, mean_RMSE, std_RMSE, mean_PICP, std_PICP, mean_MPIW, std_MPIW]
		stats.append(stat)

stats_df = pd.DataFrame(stats, columns=['var', 'slope', 'RMSE_mean', 'RMSE_std', 'PICP_mean', 'PICP_std', 'MPIW_mean', 'MPIW_std'])
stats_df.to_csv('stats.csv')