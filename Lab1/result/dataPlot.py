import csv
import matplotlib.pyplot as plt
import matplotlib as mpl
#import visdom
#vis = visdom.Visdom()

def getData(file):
	f = open(file, 'r')
	epoch = []
	accuracy = []
	loss = []
	i = 0
	bestAccuracy = 0
	for row in csv.reader(f):
		row = [float(x) for (x) in row]
		if len(row) > 0:
			if(i % 2 == 0):
				epoch.append(i / 2 + 1)
				#accuracy.append(float(row[len(row)-1]))
				accuracy.append(float(sum(row) / len(row)))
				
				if(bestAccuracy < float(sum(row) / len(row))):
					bestAccuracy = float(sum(row) / len(row))
			else:
				loss.append(float(sum(row) / len(row)))#.append(float(row[len(row)-1]))
			i = i + 1
	f.close()
	return accuracy, loss, epoch, bestAccuracy

training20Accuracy, trainin20gLoss, epoch, training20BestAccuracy = getData("./trainRes20.csv")
testing20Accuracy, testing20Loss, epoch, testing20BestAccuracy = getData("./testRes20.csv")

training56Accuracy, training56Loss, epoch, training56BestAccuracy = getData("./trainRes56.csv")
testing56Accuracy, testing56Loss, epoch, testing56BestAccuracy = getData("./testRes56.csv")

training110Accuracy, training110Loss, epoch, training110BestAccuracy = getData("./trainRes110.csv")
#testing110Accuracy, testing110Loss, epoch, testing110BestAccuracy = getData("./testRes110.csv")
testing110Accuracy, testing110Loss, epoch, testing110BestAccuracy = getData("./p100-testRes110.csv")

trainingCNN20Accuracy, trainingCNN20Loss, epoch, trainingCNN20BestAccuracy = getData("./trainCNN20.csv")
testingCNN20Accuracy, testingCNN20Loss, epoch, testingCNN20BestAccuracy = getData("./testCNN20.csv")

trainingCNN56Accuracy, trainingCNN56Loss, epoch, trainingCNN56BestAccuracy = getData("./trainCNN56.csv")
testingCNN56Accuracy, testingCNN56Loss, epoch, testingCNN56BestAccuracy = getData("./testCNN56.csv")

trainingCNN110Accuracy, trainingCNN110Loss, epoch, trainingCNN110BestAccuracy = getData("./trainCNN110.csv")
testingCNN110Accuracy, testingCNN110Loss, epoch, testingCNN110BestAccuracy = getData("./testCNN110.csv")

print len(testing20Accuracy)
print "testingRes20 accuracy:" + str(testing20BestAccuracy)
print "testingRes56 accuracy: " + str(testing56BestAccuracy)
print "testingRes110 accuracy:" + str(testing110BestAccuracy)
#print "testingCNN20 accuracy: " + str(testingCNN20BestAccuracy)
#print "testingCNN56 accuracy: " + str(testingCNN56BestAccuracy)
#print "testingCNN110 accuracy: " + str(testingCNN110BestAccuracy)


plt.xlabel("Epoch")
plt.ylabel("Accuracy")

# Residual
#plt.plot(epoch, training20Accuracy, label="training20")
plt.plot(epoch, testing20Accuracy, label="testingRes20")
#plt.plot(epoch, training56Accuracy, label="training56")
plt.plot(epoch, testing56Accuracy, label="testingRes56")
#plt.plot(epoch, training110Accuracy, label="training110")
plt.plot(epoch, testing110Accuracy, label="testingRes110")

# CNN
#plt.plot(epoch, trainingCNN20Accuracy, label="trainingCNN20")
#plt.plot(epoch, testingCNN20Accuracy, label="testingCNN20")
#plt.plot(epoch, trainingCNN56Accuracy, label="traingCNN56")
plt.plot(epoch, testingCNN56Accuracy, label="testingCNN56")
#plt.plot(epoch, trainingCNN110Accuracy, label="traingCNN110")
plt.plot(epoch, testingCNN110Accuracy, label="testingCNN110")


plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()


plt.xlabel("Epoch")
plt.ylabel("Accuracy")

# Residual Loss
#plt.plot(epoch, testing20Loss, label="testingRes20 Loss")
#plt.plot(epoch, testing56Loss, label="testingRes56 Loss")
#plt.plot(epoch, testing110Loss, label="testingRes110 Loss")

# CNN Loss
#plt.plot(epoch, testingCNN20Loss, label="testingCNN20 Loss")
plt.plot(epoch, testingCNN56Loss, label="testingCNN56 Loss")
plt.plot(epoch, testingCNN110Loss, label="testingCNN110 Loss")

plt.legend(bbox_to_anchor=(1.05, 1), loc=1, borderaxespad=0.)
plt.show()