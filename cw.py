"""
Computational Biology
protein secondary structure prediction
"""
import tensorflow as tf
import numpy as np

windowLen = 13
proTypes = 0 # number of protein type
inputLength = windowLen * proTypes
hiddenNeuron = 200
trainfilename = 'protein-secondary-structure.train'
testfilename = 'protein-secondary-structure.test'


def getpro(filename): # get protein sequences from file
	proSeq = []
	strucSeq = []
	seq = 0 # total number of protein sequence

	with open(filename, "r") as proFile: # open file
		protein = ""
		structure = ""
		for line in proFile:
			if line[0] != '<':
				protein += line[0]
				structure += line[2]
			elif line[1] == "e": # end of sequence
				seq += 1
				proSeq.append(protein)
				strucSeq.append(structure)
	return proSeq, strucSeq, seq

def getXY(i, proteinSeq, structureSeq):
	return proteinSeq[i:windowLen+i], structureSeq[(windowLen//2)+i]

def countpro(X,numofList): # count number of unique protein
	uniquepro = []
	for i in range(numofList):
		for char in X[i]:
			if char not in uniquepro:
				uniquepro.append(char)
	proTypes = len(uniquepro)
	# print uniquepro

def onehot():


"""feed forward net"""
# one-hot input for each amino acid
"""X = tf.placeholder(tf.float32, [None, inputLength])
# first hidden layer weights
w1 = tf.get_variable("w1", shape=[inputLength,hiddenNeuron], initializer=tf.contrib.layers.xavier_initializer())
# matrix multiplication to get hidden layer 1
h1 = tf.matmul(X, w1)
# rectified linear unit (ReLU nonlinearity)
h1R = tf.nn.relu(h1)

# output layer
w2 = tf.get_variable("w2", shape=[hiddenNeuron,3], initializer=tf.contrib.layers.xavier_initializer())
# matrix multiplication to get hidden layer 2
h2 = tf.matmul(h1R, w2)
# rectified linear unit (ReLU nonlinearity)
Yhat = tf.nn.relu(h2)
# predict by choosing highest type
predict = tf.argmax(Y,1)

Y = tf.placeholder(tf.float32, [None, 3]) # supervised output
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Yhat, labels=Y)) # compute costs
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost) # construct an optimizer
"""
proteinLists, structureLists, numofList = getpro(trainfilename) # load protein sequences from file
countpro(proteinLists, numofList)
"""with tf.Session() as sess:
	tf.global_variables_initializer().run()

		# sess.run(update, feed_dict={,})
	for j in range(numofList):
		l = len(proteinLists[j])-(windowLen-1)
		print 'sequence ' + str(j+1)
		for i in range(l):
			trainX, trainY = getXY(i,proteinLists[j],structureLists[j])
			print trainX, trainY"""
