import os
import pandas as pd


model_dic = {
	'PyroMLEClassify': '1FC',
	'PyroDeepMLEClassify': '3FC',
	'PyroConvClassify': 'Conv'
}

path = 'logs'
results = []

best_epoch = None
for var in os.listdir(path):
	var_path = os.path.join(path, var)
	for data in os.listdir(var_path):
		data_path = os.path.join(var_path, data)
		for model in os.listdir(data_path):
			model_path = os.path.join(data_path, model)
			for slope in os.listdir(model_path):
				slope_path = os.path.join(model_path, slope)
				for seed in os.listdir(slope_path):
					seed_path = os.path.join(slope_path, seed)
					
					epoch_path = os.path.join(seed_path, 'best_epoch.txt')
					with open(epoch_path, 'r') as f:
						best_epoch = f.readline()
						
					res_path = os.path.join(seed_path, 'results.txt')
					with open(res_path, 'r') as f:
						lines = f.readlines()
						tab_lines = [line.split(' ') for line in lines]

						metric_values = []
						# print(model, data)

						for line in tab_lines:
							if line[3] == best_epoch and line[1] == 'test':
								metric_value = round(float(line[7][:-1]), 3)

								metric_values.append(metric_value)

						res = [var, data, model_dic[model], slope[29:-1], seed] + metric_values
						results.append(res)

df = pd.DataFrame(results, columns=['var', 'data', 'model', 'slope', 'seed', 'ECE', 'Accuracy', 'NLL'])
df = df[df['seed'] <= '5']
df.to_csv('results.csv')

