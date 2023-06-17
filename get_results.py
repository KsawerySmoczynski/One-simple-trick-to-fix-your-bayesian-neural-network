import os
import pandas as pd


path = 'logs'
results = []

best_epoch = None
for var in os.listdir(path):
	var_path = os.path.join(path, var)
	for slope in os.listdir(var_path):
		slope_path = os.path.join(var_path, slope)
		for seed in os.listdir(slope_path):
			seed_path = os.path.join(slope_path, seed)
			
			epoch_path = os.path.join(seed_path, 'best_epoch.txt')
			
			# print(slope_path)
			# print(seed_path)
			# print(epoch_path)

			with open(epoch_path, 'r') as f:
				best_epoch = f.readline()
				
			res_path = os.path.join(seed_path, 'results.txt')
			with open(res_path, 'r') as f:
				lines = f.readlines()
				tab_lines = [line.split(' ') for line in lines]

				metric_values = []

				for (idx, line) in enumerate(tab_lines):
					if line[1] == str(int(best_epoch) + 1):
						for i in range(3):
							metric_value = float(tab_lines[idx + i + 4][2][:-1])
							metric_values.append(metric_value)
						break

				res = [var, slope[29:-1], seed] + metric_values
				results.append(res)

df = pd.DataFrame(results, columns=['var', 'slope', 'seed', 'RMSE', 'PICP', 'MPIW'])
df.to_csv('results.csv')