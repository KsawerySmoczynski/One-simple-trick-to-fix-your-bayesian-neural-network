import pandas as pd

auto_df = pd.read_csv('auto_stats.csv')
manual_df = pd.read_csv('manual_stats.csv')

for data in auto_df['data'].unique():
	for model in auto_df['model'].unique():
		tauto_df = auto_df[(auto_df['model'] == model) & (auto_df['data'] == data) & (auto_df['slope'] == -1.0)]
		tmanual_df = manual_df[(manual_df['model'] == model) & (manual_df['data'] == data) & (auto_df['slope'] == 0.0)]
		
		auto, manual = float(tauto_df['ECE_mean']), float(tmanual_df['ECE_mean'])

		print(data, model)
		print(f"auto: {auto}, manual: {manual}, ratio: {auto / manual}")
