import re, sys, os
from collections import Counter
pPath = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(pPath)
import readFasta
import checkFasta
import numpy as np
import pandas as pd

USAGE = """
USAGE:
	python EAAC.py input.fasta <sliding_window> <output>

	input.fasta:      the input protein sequence file in fasta format.
	sliding_window:   the sliding window, integer, defaule: 5
	output:           the encoding file, default: 'encodings.tsv'
	order:            the out order, select from ['alphabetically', 'polarity', 'sideChainVolume' or userDefined] 
"""

def EAAC(fastas, window=5, **kw):
	if checkFasta.checkFasta(fastas) == False:
		print('Error: for "EAAC" encoding, the input fasta sequences should be with equal length. \n\n')
		return 0

	if window < 1:
		print('Error: the sliding window should be greater than zero' + '\n\n')
		return 0

	if checkFasta.minSequenceLength(fastas) < window:
		print('Error: all the sequence length should be larger than the sliding window :' + str(window) + '\n\n')
		return 0

	AA = kw['order'] if kw['order'] != None else 'ACDEFGHIKLMNPQRSTVWY'
	#AA = 'ARNDCQEGHILKMFPSTWYV'
	encodings = []
	header = ['#']
	for w in range(1, len(fastas[0][1]) - window + 2):
		for aa in AA:
			header.append('SW.'+str(w)+'.'+aa)
	encodings.append(header)

	for i in fastas:
		name, sequence = i[0], i[1]
		code = [name]
		for j in range(len(sequence)):
			if j < len(sequence) and j + window <= len(sequence):
				count = Counter(re.sub('-', '', sequence[j:j+window]))
				for key in count:
					count[key] = count[key] / len(re.sub('-', '', sequence[j:j+window]))
				for aa in AA:
					code.append(count[aa])
		encodings.append(code)
	return encodings

kw = {'path': r"H_train.txt",'order': 'ACDEFGHIKLMNPQRSTVWY'}

fastas = readFasta.readFasta(r"H_train.txt")

sw=5
result = EAAC(fastas, sw, **kw)



data1=np.matrix(result[1:])[:,1:]
data_=pd.DataFrame(data=data1)
data_.to_csv('EAAC_H_train.csv')
