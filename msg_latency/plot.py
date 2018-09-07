import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pprint


def yield_data():
	deviation = list()
	average = list()
	label = list()
	first = True
	f = open("./time.dat", "r")
	for line in f.readlines():
		if line.startswith("[size]"):
			if not first:
				yield average, deviation, label
			first = False
			average = list()
			deviation = list()
			label = list()
			continue
		raw = line.split("\t")
		label.append(raw[0])
		average.append(float(raw[1]))
		deviation.append(float(raw[2]))
	yield average, deviation, label


if __name__ == '__main__':
	fig, ax = plt.subplots()
	bar_width = 0.5
	opacity = 1.0
	color = ['r', 'b', 'g', 'lime']
	error_config = {'ecolor': '0.3'}
	pair = 0

	for average, deviation, label in yield_data():
		index = [3*loc+pair*bar_width for loc in range(len(label))]
		rects1 = ax.bar(index, average, bar_width, alpha=opacity, 
			yerr=deviation, error_kw=error_config, color=color[pair], 
			label='Pair%d' % (pair))
		pair += 1

	ax.set_xlabel('Size')
	ax.set_ylabel('Time')
	ax.set_xticks([3*loc+pair/2*bar_width for loc in range(len(label))])
	ax.set_xticklabels(label)
	ax.legend()

	fig.tight_layout()
	plt.show()






