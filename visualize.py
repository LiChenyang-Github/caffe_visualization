#coding:utf-8

import numpy as np
import matplotlib.pyplot as plt
import sys,os,caffe

# ***************************
# the direction and params you need to tell the code 
# ***************************
caffe_root = '/home/lichenyang/gesture_caffe_1/caffe/' 
modelDir = 'examples/third_gesture/experiment/5/result/_iter_100000.caffemodel'
prototxtDir = 'examples/third_gesture/deploy.prototxt'
imageDir = 'examples/third_gesture/data/val/cai_qiuxia/cai_qiuxia_flip110.jpg'
save_path = '/home/lichenyang/gesture_caffe_1/caffe/examples/third_gesture/visualization/'
IMGHEIGHT = 480
IMGWIDTH = 640
# Attention: if you use meanfile, you should write a func to change the meanfile into python mode
MEANVALUE = 127 

sys.path.insert(0, caffe_root + 'python')
os.chdir(caffe_root)
if not os.path.isfile(caffe_root + modelDir):
    print("caffemodel is not exist...")

caffe.set_mode_gpu()
net = caffe.Net(caffe_root + prototxtDir,
                caffe_root + modelDir,
                caffe.TEST)



# use transformer to change the img into a appropriate form as the network input
# and the following code is to initialize the transformer
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
# python load img as H×W×K，we need it to be K×H×W
transformer.set_transpose('data', (2,0,1))
mean = np.zeros((3, IMGHEIGHT, IMGWIDTH))   # in BGR order
mean[:,:,:] = MEANVALUE
# mean pixel
transformer.set_mean('data', mean) 
# the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_raw_scale('data', 255)  
# the reference model has channels in BGR order instead of RGB
transformer.set_channel_swap('data', (2,1,0))  

# resize the data with reshape
# TO BE MODIFIED!!!
im = caffe.io.load_image(imageDir)
# reshape the inout of the network as it in the deploy.prototxt
net.blobs['data'].reshape(1,3,IMGHEIGHT,IMGWIDTH)
# use transformer to deal witn the img 
net.blobs['data'].data[...] = transformer.preprocess('data', im)
out = net.forward()

# take an array of shape (n, height, width) or (n, height, width, channels)
# and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
def vis_square(data, padsize=1, padval=0):
    # do normalization
    data -= data.min()
    data /= data.max()
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    
    plt.imshow(data)
    plt.axis('off')

# ***************************
# print the shape of blob and param(including w and b) of each layer
# ***************************
# print [(k, v.data.shape) for k, v in net.blobs.items()] # the info of feature map of each layer (C,H,W)
# print [(k, v[0].data.shape) for k, v in net.params.items()] # the shape of w matrix of each layer
# print [(k, v[1].data.shape) for k, v in net.params.items()] # the shape od b matrix of each layer



# --------------------------------------------------------------------------------
# the following code is to draw and save some pictures, you need to change some 
# params according to your deploy.txt and the shape of the input img
# --------------------------------------------------------------------------------

# ***************************
# the parameters are a list of [weights, biases]
# ***************************
filters = net.params['conv1'][0].data
vis_square(filters.transpose(0, 2, 3, 1))
plt.savefig(save_path + 'conv1_params.jpg')

filters = net.params['conv2'][0].data
#vis_square(filters.reshape(48*96, 5, 5))
# just visulize the first 48 channels among 96
vis_square(filters[:48].reshape(48**2, 5, 5))
plt.savefig(save_path + 'conv2_params.jpg')


# ***************************
# feat is the blob of each conv layer or pool layer
# data[0, :48] mean : use the first img and the first 
# 48th channel, net.blobs['conv1'].data.shape is (1,48,118,158)
# ***************************
feat = net.blobs['conv1'].data[0, :48]
vis_square(feat, padval=1)
plt.savefig(save_path + 'conv1_blobs.jpg')

feat = net.blobs['conv2'].data[0, :96]
vis_square(feat, padval=1)
plt.savefig(save_path + 'conv2_blobs.jpg')

feat = net.blobs['conv3'].data[0]
vis_square(feat, padval=0.5)
plt.savefig(save_path + 'conv3_blobs.jpg')

feat = net.blobs['conv4'].data[0]
vis_square(feat, padval=0.5)
plt.savefig(save_path + 'conv4_blobs.jpg')

feat = net.blobs['conv5'].data[0]
vis_square(feat, padval=0.5)
plt.savefig(save_path + 'conv5_blobs.jpg')


# ***************************
# full connect
# ***************************
feat = net.blobs['fc6'].data[0]
plt.subplot(2, 1, 1)
plt.plot(feat.flat)
plt.savefig(save_path + 'fc6.jpg')
plt.subplot(2, 1, 2)
_ = plt.hist(feat.flat[feat.flat > 0], bins=100)
plt.savefig(save_path + 'fc6_hist.jpg')

feat = net.blobs['fc7'].data[0]
plt.subplot(2, 1, 1)
plt.plot(feat.flat)
plt.savefig(save_path + 'fc7.jpg')
plt.subplot(2, 1, 2)
_ = plt.hist(feat.flat[feat.flat > 0], bins=100)
plt.savefig(save_path + 'fc7_hist.jpg')

feat = net.blobs['fc8'].data[0]
plt.subplot(2, 1, 1)
plt.plot(feat.flat)
plt.savefig(save_path + 'fc8.jpg')
plt.subplot(2, 1, 2)
_ = plt.hist(feat.flat[feat.flat > 0], bins=100)
plt.savefig(save_path + 'fc8_hist.jpg')

