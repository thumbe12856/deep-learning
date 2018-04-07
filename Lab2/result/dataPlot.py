import csv
import matplotlib.pyplot as plt
import matplotlib as mpl
#import visdom
#vis = visdom.Visdom()

def getData(file):
	f = open(file, 'r')
	epoch = []
	loss = []
	i = 0
	bestAccuracy = 0
	for row in csv.reader(f):
		row = [float(x) for (x) in row]
		loss = row
	f.close()
	return loss

img_loss = getData("./old/r1_img.csv")
img_with_noise = getData("./old/r1_img_with_noise.csv")
r1_shuffle1 = getData("./old/r1_shuffle.csv")
#r1_shuffle = getData("./r1_shuffle.csv")
r1_only_noise = getData("./old/r1_only_noise.csv")

epoch = [i for i in range(len(img_loss))]

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 14}
plt.rc('font', **font)
plt.xlabel("Iteration (log scale)")
plt.ylabel("MSE")
plt.xscale('log')

plt.plot(epoch, img_loss, label="Image")
plt.plot(epoch, img_with_noise, label="Image + noise")
plt.plot(epoch, r1_shuffle1, label="Image shuffled")
plt.plot(epoch, r1_only_noise, label="U(0, 1) noise")

plt.legend(bbox_to_anchor=(1.05, 1), loc=1, borderaxespad=0.)
plt.show()