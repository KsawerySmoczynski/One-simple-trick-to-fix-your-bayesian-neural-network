import pandas as pd
import numpy as np

LABEL = 'auto-label'
CAPTION = 'auto-caption'
SCALE = '1'
LEAKY_SLOPE = -0.3
FUN = 'auto_class'
metrics = ["Acc", "ECE", "NLL"]


def show_metric(mean, std, better):
	if better:
		return "& \\textbf{" + mean + " $\\pm$ " + std + "} "
	else:
		return "& " + mean+ " $\\pm$ " + std + " "

def cmp(m1, m2, metric):
	m1, m2 = float(m1), float(m2)
	if metric in ['ECE', 'NLL']:
		return m1 <= m2
	else:
		return m1 >= m2


def auto_class():
	mdf = pd.read_csv('manual_stats.csv')

	rel_df = mdf[mdf['slope'] == 0.0]
	lea_df = mdf[mdf['slope'] == LEAKY_SLOPE]

	return rel_df, lea_df

def auto_manual():
	rel_df = pd.read_csv('manual_stats.csv')
	lea_df = pd.read_csv('auto_stats.csv')

	rel_df = rel_df[rel_df['slope'] == 0.0]
	lea_df = lea_df[lea_df['slope'] == LEAKY_SLOPE]

	return rel_df, lea_df

FUN_DIC = {'auto_class': auto_class, 'auto_manual': auto_manual}


def make_tab(rel_df, lea_df):
	tab = "\\begin{table}[b!]\n"
	tab += "\\label{"+LABEL+"}\n"
	tab += "\\centering\n"
	tab += "\\caption{"+CAPTION+"}\n"
	tab += "\\scalebox{"+SCALE+"}{\n"
	tab += "\\begin{tabular}{ll"
	for _ in metrics:
		tab += "cc"
	tab += "}\n"
	tab += "\\toprule\n"
	tab += " & {} "

	for m in metrics:
		tab += "& \\multicolumn{2}{c}{"+m+"} "

	tab += "\\\\\n &  "

	for _ in metrics:
		tab += "& Leaky ReLU & ReLU "

	tab += "\\\\\n"

	tab += "Dataset & Model & "

	for m in metrics:
		tab += "& "

	tab += "\\\\\n"
	tab += "\\midrule\n"

	for data in rel_df['data'].unique():
		tab += data
		for model in rel_df['model'].unique():
			tab += " & "
			tab += model
			tab += " "

			for m in metrics:
				rdf = rel_df[(rel_df['model'] == model) & (rel_df['data'] == data)]
				ldf = lea_df[(lea_df['model'] == model) & (lea_df['data'] == data)]
				
				mean_l = str(float(ldf[m+"_mean"]))
				std_l = str(float(ldf[m+"_std"]))

				mean_r = str(float(rdf[m+"_mean"]))
				std_r = str(float(rdf[m+"_std"]))

				# if m == 'NLL':
				# 	print(mean_l)
				# 	print(mean_r)
				# 	print(cmp(mean_l, mean_r, m))

				tab += show_metric(mean_l, std_l, cmp(mean_l, mean_r, m))
				tab += show_metric(mean_r, std_r, cmp(mean_r, mean_l, m))

			tab += "\\\\\n"

	tab += "\n\\bottomrule\n"
	tab += "\\end{tabular}\n}\n"
	tab += "\\end{table}\n"
	return tab

# print(FUN_DIC[FUN]())

rel_df, lea_df = FUN_DIC[FUN]()
tab = make_tab(rel_df, lea_df)

with open('tex_table.txt', 'w') as f:
	f.write(tab)

print(tab)