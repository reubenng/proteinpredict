"""
Computational Biology
protein secondary structure prediction
"""
import tensorflow as tf
import numpy as np

windowLen = 13
proTypes = 20 # number of protein type
inputLength = windowLen * proTypes
hiddenNeuron = 200
trainfilename = 'protein-secondary-structure.train'
testfilename = 'protein-secondary-structure.test'
structype = ['_', 'e', 'h']
epoch = 10

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
	# print uniquepro
	return uniquepro

def protonum(trainX, trainY, uniquepro):
	num = [None]* windowLen
	stype = [None]* windowLen
	for i in range(len(uniquepro)):
		for j in range(windowLen):
			if trainX[j] == uniquepro[i]:
				num[j] = i
	for k in range(3):
		if trainY == structype[k]:
			stype = k
	# print stype
	num = np.array(num)
	num = num.reshape([1,13])
	stype = np.array(stype)
	stype = stype.reshape([1,1])
	return num, stype

"""feed forward net"""
# one-hot input for each amino acid
X = tf.placeholder(tf.int32, [None, windowLen])
onehotIn = tf.one_hot(X, 20)
onehotR = tf.reshape(onehotIn, [-1,20*13])
# first hidden layer weights
w1 = tf.get_variable("w1", shape=[inputLength,hiddenNeuron], initializer=tf.contrib.layers.xavier_initializer())
# matrix multiplication to get hidden layer 1
h1 = tf.matmul(onehotR, w1)
h1R = tf.nn.sigmoid(h1)

# output layer
w2 = tf.get_variable("w2", shape=[hiddenNeuron,3], initializer=tf.contrib.layers.xavier_initializer())
# matrix multiplication to get hidden layer 2
# h2 = tf.matmul(h1R, w2)
Yhat = tf.matmul(h1R, w2)
# Yhat = tf.nn.sigmoid(h2)
# predict by choosing highest type
predict = tf.argmax(Yhat,1)

Y = tf.placeholder(tf.int32, [None, 1]) # supervised output
onehotOut = tf.one_hot(Y, 3)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Yhat, labels=onehotOut)) # compute costs
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost) # construct an optimizer

correct_prediction = tf.equal(tf.argmax(onehotOut, 1), tf.argmax(Yhat, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
	tf.global_variables_initializer().run()

	proteinLists, structureLists, numofList = getpro(trainfilename) # load protein sequences from file
	uniquepro = countpro(proteinLists, numofList)

	# train
	for r in range(epoch):
		print 'Training epoch ' + str(r)
		for j in range(numofList):
			l = len(proteinLists[j])-(windowLen-1)
			print 'Sequence ' + str(j+1)
			for i in range(l):
				trainX, trainY = getXY(i,proteinLists[j],structureLists[j])
				# print trainX, trainY
				x, y = protonum(trainX, trainY, uniquepro)
				sess.run(train_op, feed_dict={X:x, Y:y})
		print 'Training done.'

	# test
	a = 0.0
	print 'Testing...'
	eproteinLists, estructureLists, enumofList = getpro(testfilename) # load protein sequences from file
	for j in range(enumofList):
		l = len(eproteinLists[j])-(windowLen-1)
		print 'Sequence ' + str(j+1)
		for i in range(l):
			testX, testY = getXY(i,eproteinLists[j],estructureLists[j])
			# print trainX, trainY
			xe, ye = protonum(testX, testY, uniquepro)
			# print(sess.run(accuracy, feed_dict={X:xe, Y:ye}))
			print testX
			prediction = structype[sess.run(predict, feed_dict={X:xe})[0]]
			print 'Prediction: ' + str(prediction)
			print 'Actual: ' + str(testY)
			if prediction == testY:
				a += 1
	print 'Accuracy: ' + str((a/(l*enumofList))*100)