import cPickle as pkl
import numpy

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import skimage
import skimage.transform
import skimage.io
from PIL import Image
import csv

img = skimage.io.imread("test.png")
n_words = 9
words = ["a", ",b", ",c"]#, ",d", ",e", ",f", ",g", ",h", ",i", "j", "a", ",b", ",c", ",d", ",e", ",f", ",g", ",h", ",i", "j", "a", ",b", ",c", ",d", ",e", ",f", ",g", ",h", ",i", "j", "a", ",b", ",c", ",d", ",e", ",f", ",g", ",h", ",i", "j", "a", ",b", ",c", ",d", ",e", ",f", ",g", ",h", ",i", "j"]
n_alpha = 3

for i in range(n_alpha):
    files = 'alpha' + str(i) + '.csv'
    f = open(files, 'r')
    j = 0
    for row in csv.reader(f):
        row = map(float, row)
        if j == 0:
            temp = numpy.array([row])
        else:
            temp = numpy.append(temp, [row], axis=0)
        j = j + 1

    if(i == 0):
        alpha = numpy.array([temp])
    else:
        alpha = numpy.append(alpha, numpy.array([temp]), axis=0)
    f.close()
print alpha.shape

#assert False

# display the visualization
n_words = alpha.shape[0] + 1
w = numpy.round(numpy.sqrt(n_words))
h = numpy.ceil(numpy.float32(n_words) / w)
        
plt.subplot(w, h, 1)
plt.imshow(img)
plt.axis('off')

smooth = True

for ii in xrange(alpha.shape[0]):
    plt.subplot(w, h, ii+2)
    lab = words[ii]
    plt.text(0, 1, lab, backgroundcolor='white', fontsize=13)
    plt.text(0, 1, lab, color='black', fontsize=13)
    plt.imshow(img)
    if smooth:
        alpha_img = skimage.transform.pyramid_expand(alpha[ii, 0,:].reshape(14,14), upscale=16, sigma=20)
    else:
        alpha_img = skimage.transform.resize(alpha[ii, 0,:].reshape(14,14), [img.shape[0], img.shape[1]])
    plt.imshow(alpha_img, alpha=0.8)
    plt.set_cmap(cm.Greys_r)
    plt.axis('off')
plt.show()
