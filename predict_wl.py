import chainer
from PIL import Image
import numpy as np
import argparse
import glob
import chainer
import chainer.functions as F
import chainer.links as L
from chainer.backends import cuda 
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
from chainer import training
from chainer.training import extensions

from chainer import serializers

use_gpu = True

class MLP(chainer.Chain):

    def __init__(self, n_units1, n_units2):
        super(MLP, self).__init__()
        with self.init_scope():
            initializer = chainer.initializers.HeNormal()
            self.l1 = L.Convolution2D(3, n_units1, 5, stride=1, pad=2, initialW=initializer)
            #self.l2 = L.Convolution2D(None, n_units2, 5, stride=1, pad=2, initialW=initializer)
            self.l3 = L.Convolution2D(None, 1, 1)

    def __call__(self, *args):

        #assert len(args) >= 2
        x = args[0]
        #t = args[-1]
        #t = x
        #self.y = None
        #self.loss = None
        #self.accuracy = None
        self.y = self.forward(x)
        #self.loss = self.lossfun(self.y, t)
        #chainer.report({'loss': self.loss}, self)
        #print(np.sqrt(mean_squared_error(t[0][0], self.y.data[0][0])))
        return self.y

    def forward(self, x):
        #print(x.data.shape)
        h1 = F.relu(self.l1(x))
        #h2 = F.relu(self.l2(h1))
        #print(h1.data.shape)
        h3 = self.l3(h1)
        return h3

    def lossfun(self, y, t):
        #return F.sigmoid_cross_entropy(y,t)
        return (y * (t - (y >= 0)) - F.log1p(F.exp(-F.abs(y))))
        #return chainer.functions.mean_squared_error(F.tanh(y), (-t*2+1).astype(np.float32))


def predict(img):
    #print(img)
    a = np.asarray(img).transpose(2,0,1).astype(np.float32)/255.
    x = np.expand_dims(a,axis=0)
    if use_gpu:
        y = model(cuda.to_gpu(x))
    else:
        y = model(x)

    ret = y.data[0][0] > -3
    if use_gpu:
        ret = cuda.to_cpu(ret)
    return ret.astype(np.int32)

def predict_test():
    #print('predict_test')
    fname = '000003.jpg'
    imw = 320
    imh = 240
    img = Image.open(fname).convert('RGB').resize((imw,imh))
    y = predict(np.asarray(img))
    return y

model = MLP(4,2)
serializers.load_npz("mymodel0617.npz", model)
if use_gpu:
    model.to_gpu()

if False and __name__ == '__main__':
    result_im = predict_test()
    print(result_im)
    thimg = (result_im > 0)
    thimg = thimg.astype(np.uint8) * 255
    thimg = Image.fromarray(thimg, 'L')
    plt.imshow(thimg)
    plt.pause(3)
