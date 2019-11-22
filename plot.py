import matplotlib.pyplot as plt

def plot(model_name = "RoBERTa_v2_Experiments", layers = None, exp = None):

	for name, experiment in exp.items():
		for metric, val in experiment.items():
			for params, stats in val.items():
				looper_dropout = params
				plt.plot(layers, stats, label= str(name) + ", " + str(metric) + ", " + "looper_dropout=" + str(looper_dropout))
	
	plt.xlabel('Numbers of decoder layers')
	plt.ylabel('Scores')
	plt.title(model_name)
	plt.legend(loc=0, prop={'size': 6})
	plt.savefig("plots/" + model_name + ".png")

if __name__ == '__main__':
	
	layers = [1, 2, 4, 6, 8, 10, 12]
	
	# key: (dataset_name, looper_dropout)
	f1_base = {(0.0): [0.825397, 0.833333, 0.824903, 0.813278, 0.803213, 0.823529, 0.829457],
	(0.3): [0.826389, 0.856061, 0.845283, 0.84375, 0.835938, 0.815385, 0.844961]}
	acc_base = {(0.0): [0.891827, 0.894231, 0.889423, 0.889423, 0.882212, 0.889423, 0.891827],
	(0.3): [0.887019, 0.90625, 0.899038, 0.901442, 0.896635, 0.882212, 0.901442]}
	f1_fg = {(0.0): [0.828897, 0.831373, 0.826446, 0.820896, 0.818898, 0.832685, 0.822581],
	(0.2): [0.836653, 0.853933, 0.830769, 0.823077, 0.832714, 0.823529, 0.837945]}
	acc_fg = {(0.0): [0.894231, 0.894231, 0.894231, 0.894231, 0.887019, 0.894231, 0.891827],
	(0.2): [0.899038, 0.903846, 0.891827, 0.887019, 0.891827, 0.894231, 0.899038]}
	
	# exp
	exp = {}
	exp["Baseline_MRPC"] = {"f1": f1_base, "acc": acc_base}
	exp["Forget_MRPC"] = {"f1": f1_fg, "acc": acc_fg}
	plot("RoBERTa_v2_Experiments", layers, exp)