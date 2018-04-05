import skimage.measure
from scipy import misc

groundTruthImg = misc.imread("./result/2-in.png")
testImg = misc.imread("./result/2-out.png")

psnr = skimage.measure.compare_psnr(groundTruthImg, testImg)
print psnr
