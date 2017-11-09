
# coding: utf-8

# In[1]:


from keras.layers import Concatenate, Activation, Lambda, Subtract, Dot, Multiply
from keras.layers import Input, Dense, Flatten, Convolution1D, RepeatVector, Add
from keras.layers import Dropout
from keras.engine.topology import Container
from keras.models import Model, Sequential
from keras.initializers import Zeros
from keras.applications.vgg19 import VGG19
from keras.activations import softmax
from keras import backend as K
from keras import optimizers
from keras.callbacks import ModelCheckpoint
import sys, glob, math, os
import cv2
import numpy as np
import re
from numpy.random import *

ncluster = 50
nfeature = 196
nfeature_dim = 512
nbatch = 2


# In[ ]:


def make_data(data,centers,gt_label):

    idx = np.random.randint(0,data.shape[0],data.shape[0])

    data_all = data
    data_all2 = data[idx,:,:]
    gt_label_all = gt_label
    gt_label_all2 = gt_label[idx]
    centers_all = centers

    for i in range(0,60):
        data_all = np.concatenate((data_all,data),axis=0)
        idx = np.random.randint(0,data.shape[0],data.shape[0])
        data_all2 = np.concatenate((data_all2,data[idx,:,:]),axis=0)
        gt_label_all  = np.concatenate((gt_label_all,gt_label))
        gt_label_all2 = np.concatenate((gt_label_all2,gt_label[idx]))
        centers_all = np.concatenate((centers_all,centers), axis=0)
        idx = np.random.randint(0,data.shape[0],data.shape[0])
          
    # ground truth
    y_gt = np.zeros((data_all.shape[0]))

    for i in range(0,gt_label_all.shape[0]):
        if gt_label_all[i] == gt_label_all2[i]:
            y_gt[i] = 1
        else:
            y_gt[i] = 0
        
    # choose positives and negatives as same samples
    p_idx = []
    for i in range(0,gt_label_all.shape[0]):
        if gt_label_all[i] == gt_label_all2[i]:
            p_idx.append(i)

    n_idx = []
    for i in range(0,gt_label_all.shape[0]):
        if gt_label_all[i] != gt_label_all2[i]:
            n_idx.append(i)

    idx = p_idx
    idx.extend(n_idx[0:len(p_idx)])

    data_all = data_all[idx,:,:]
    data_all2 = data_all2[idx,:,:]
    centers_all = centers_all[idx,:,:]
    y_gt = y_gt[idx]
    
    return data_all, data_all2, centers_all, y_gt


# In[2]:


def custom_softmax(x):
    y = softmax(x, axis=1)
    return y


# In[3]:


def transpose(x):
    x = K.permute_dimensions(x,(1,2,0))
    y = K.transpose(x)
    return y


# In[4]:


def make_feature_tensor(x,i,c):
    # make [xi-c(1) xi-c(2) ... xi-c(n)] tensor
    t = x[:,i,:]
    z = K.repeat(t, ncluster)
    z = z - c
#    z = K.permute_dimensions(z,(0,1,2))
    return z


# In[5]:


def rep_tensor_and_mult(w,x,c,batch_size):
    zz = K.zeros(shape=(batch_size,nfeature_dim,ncluster))
    
    for i in range(0,nfeature):
        # Feature vector
        t = x[:,i,:]
        t = K.repeat(t, ncluster)
        t = t - c
        t = K.permute_dimensions(t,(1,2,0))
        t = K.transpose(t)
        
        # Weight vector
        ww = w[:,i,:]
        z = K.repeat(ww, nfeature_dim)
                
        tmp = z*t
        zz = zz + tmp
        # K.update_add(zz,tmp)
        #zz = zz + tmp
        #zz = K.update_add(zz,tmp)
    
    return zz


# In[16]:


def generator_loss(y_true, y_pred): # y_true's shape=(batch_size, row, col, ch)
    S = ncluster*nfeature_dim
    x1 = y_pred[:,:S]
    x2 = y_pred[:,S:]
    z = K.batch_dot(x1,x2,axes=1)
    z = K.abs(y_true - K.relu(z))
    return z


# In[17]:


def Split_and_dot(x):
    s = K.int_shape(x)
    x1 = x[:,:25600]
    x2 = x[:,25600:]
    
    z = K.batch_dot(x1,x2,axes=1)
    
    return z


# In[18]:


def L1_normalize(x):
    #x = K.l2_normalize(x, axis=1)
    x = x / K.sum(x, axis=1, keepdims=True)
    return x


# In[19]:


def L2_normalize(x):
    x = K.l2_normalize(x, axis=1)
    #x = x / K.sum(x, axis=1, keepdims=True)
    return x


# In[20]:


def netvlad_model():
    # input feature and cluster centers
    x_input  = Input([nfeature,nfeature_dim], name='x_input')
    x_input2 = Input([nfeature,nfeature_dim], name='x_input2') 
    c_input = Input([ncluster,nfeature_dim], name='c_input')
    
    # start designing BASE_LAYERS
    conv_1 = []
    for i in range(0,ncluster):
        x = Convolution1D(1, 1, padding='same', use_bias=True)(x_input)
#        x = Dropout(0.5)(x)
        conv_1.append(x)
    
    w = Concatenate(axis=2)(conv_1)
    w = Lambda(custom_softmax, name='cumstom_softmax')(w)
    z = Lambda(rep_tensor_and_mult, arguments={'x': x_input,'c': c_input,'batch_size': nbatch}, name='rep_tensor_mult')(w)
    z = Flatten()(z)
    z = Lambda(L2_normalize)(z)
    # end BASE_LAYERS
    
    # define second reference layers
    base_layers = Container(x_input, z, name="base_layers")
    z2 = base_layers(x_input2)
    
    # concatenate in one list
    z_list = []    
    z_list.append(z)
    z_list.append(z2)
    
    z_out = Concatenate()(z_list)
    #z_out = Lambda(Split_and_dot)(z_out)
    
    model = Model(inputs=[x_input,x_input2,c_input], outputs=z_out)
    
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss=generator_loss)
    
#    model.summary()
 
    return model
#END vgg19_lower_model()


# In[21]:


model = netvlad_model()


# In[3]:


data = np.load("data.npy")
center = np.load("centers.npy")
centers = np.zeros((data.shape[0],center.shape[0],center.shape[1]))
for i in range(0,data.shape[0]):
    centers[i,:,:]=center

me = np.load("mean.npy")
stv = np.load("stv.npy")
gt_label = np.loadtxt("result-frame-org.csv",delimiter=",", usecols=(1))


# In[4]:


print(data.shape)
print(centers.shape)
print(me.shape)
print(stv.shape)


# In[14]:


t_data_all, t_data_all2, t_centers_all, t_y_gt = make_data(data,centers,gt_label)
v_data_all, v_data_all2, v_centers_all, v_y_gt = make_data(data,centers,gt_label)


# In[13]:


#cp = ModelCheckpoint('./cache/model_weights_{epoch:02d}.h5')

model.fit([t_data_all,t_data_all2,t_centers_all], t_y_gt, 
          batch_size=nbatch, 
          epochs=100, 
          verbose=1, 
          validation_data=([v_data_all,v_data_all2,v_centers_all], v_y_gt),
          shuffle=True )
#          callbacks=[cp])

# In[ ]:


#model.fit([data_all,data_all2,centers_all], y_gt, batch_size=nbatch, epochs=20, verbose=1, shuffle=True)


# In[58]:


model.save_weights('netvlad2-weights.h5')

