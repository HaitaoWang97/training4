#!/usr/bin/env python
# coding: utf-8

# In[4]:


from mxnet import nd
from mxnet.gluon import nn

def pool2d(X, pool_size, mode='max'):
    p_h , p_w = pool_size
    Y = nd.zeros((X.shape[0] - p_h + 1,X.shape[1] - p_w +1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i: i + p_h, j: j + p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i: i + p_h, j: j + p_w].mean()
    return Y


# In[5]:


X = nd.array([[0,1,2],[3,4,5],[6,7,8]])
pool2d(X, (2,2))


# In[6]:


pool2d(X, (2,2),'avg')


# In[ ]:




