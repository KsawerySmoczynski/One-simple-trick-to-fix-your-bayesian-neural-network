import pandas as pd

df = pd.read_csv('results.csv')
stats = []

for var in df['var'].unique():
	for data in df['data'].unique():
		for model in df['model'].unique():
			for slope in df['slope'].unique():
				tdf = df[(df['var'] == var) & (df['data'] == data) & (df['model'] == model) & (df['slope'] == slope)]
				
				mean_ece = round(tdf['ECE'].mean(), 3)
				std_ece = round(tdf['ECE'].std(), 3)

				mean_acc = round(tdf['Accuracy'].mean(), 3)
				std_acc = round(tdf['Accuracy'].std(), 3)

				mean_nll = round(tdf['NLL'].mean(), 3)
				std_nll = round(tdf['NLL'].std(), 3)

				# if var == 'manual' and data == 'MNIST' and model == 'Conv' and slope == -1.0:
				# 	print(mean_nll)
				# 	print(tdf)

				stat = [var, data, model, slope, mean_ece, std_ece, mean_acc, std_acc, mean_nll, std_nll]
				stats.append(stat)

stats_df = pd.DataFrame(stats, columns=['var', 'data', 'model', 'slope', 'ECE_mean', 'ECE_std', 'Acc_mean', 'Acc_std', 'NLL_mean', 'NLL_std'])
stats_df = stats_df[stats_df['var'] == 'manual']

stats_df.to_csv('manual_stats.csv')