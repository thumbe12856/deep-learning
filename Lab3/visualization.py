import cPickle as pkl
import numpy

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import skimage
import skimage.transform
import skimage.io
from PIL import Image
import csv
import os, sys

#imageFile = "acoustic-guitar-player.jpg"
#imageFile = "COCO_val2014_000000000764.jpg"

#imageFile = "COCO_val2014_000000002191.jpg"
imageFile = './data/my_test/COCO_val2014_000000019444.jpg'

size = 224, 224
outfile = os.path.splitext(imageFile)[0] + ".thumbnail"
im = Image.open(imageFile)
im = im.resize(size, Image.LANCZOS)
im.save(outfile, "JPEG")

img = skimage.io.imread(outfile)

# a man on a tennis court holding a frisbee
words = ["a", "man", "on", "a", "tennis", "court", "holding", "a", "frisbee"]

n_words = len(words)
#alpha_index = [[0, 0], [0, 5], [2, 2], [0, 0], [4, 2], [5, 0], [2, 8], [0, 0], [4, 3]]
alpha_index = [[0, 0], [1, 0], [2, 1], [0, 0], [4, 1], [5, 0], [5, 1], [0, 0], [7, 1]]
n_alpha = n_words

alpha = numpy.array([[]])
for i in range(n_alpha):
    #index = alpha_index[i][0] + 1
    index = i+16

    files = './data/alpha/showattendtell/alpha' + str(index) + '.csv'

    #files = 'alpha' + str(i+154) + '.csv'
    f = open(files, 'r')
    j = 0
    for row in csv.reader(f):
        row = map(float, row)
        if j == 0:
            j = 1
            continue
        #elif j == alpha_index[i][1] + 1:
        elif j == 1:
            tempRow = numpy.array([])
            for r in row:
                tempRow = numpy.append(tempRow, numpy.array([r]))
            temp = numpy.array([tempRow])
            break
        j = j + 1
        #if(j == 16): break

    if(j > 0):
        if(alpha.size == 0):
            alpha = numpy.array([temp])
        else:
            alpha = numpy.concatenate((alpha, [temp]), axis=0)
    '''
    if(i == 0):
        alpha = numpy.array([temp])
    else:
        alpha = numpy.append(alpha, numpy.array([temp]), axis=0)
        '''
    f.close()

print alpha.shape

#print temp.shape
#print alpha.repeat(alpha, 1)
#print alpha.shape

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
#for ii in xrange(1):
    plt.subplot(w, h, ii+2)
    lab = words[ii]
    plt.text(0, 1, lab, backgroundcolor='white', fontsize=13)
    plt.text(0, 1, lab, color='black', fontsize=13)
    plt.imshow(img)
    if smooth:
        alpha_img = skimage.transform.pyramid_expand(alpha[ii, 0, :].reshape(14, 14), upscale=16, sigma=20)
        #alpha_img = skimage.transform.pyramid_expand(alpha, upscale=20, sigma=20)
        #alpha_img = skimage.transform.pyramid_expand(alpha[:].reshape(224, 224), upscale=16, sigma=20)
    else:
        alpha_img = skimage.transform.resize(alpha[ii, :].reshape(14,14), [img.shape[0], img.shape[1]])
    plt.imshow(alpha_img, alpha=0.8)
    plt.set_cmap(cm.Greys_r)
    plt.axis('off')
plt.show()
