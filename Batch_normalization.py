#全连接层与卷积层的批量归一化不同，卷积层需要在不同的通道
#各自归一化
import d2lzh as d2l
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import nn

def batch_norm(X, gamma,beta, moving_mean, moving_var, momentum):
     #判断当前处于何种模式
     if not autograd.is_training():
         #预测模式下直接使用训练调好的参数
         X_hat = (X - moving_mean) / nd.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            #全连接层
            mean = X.mean(axis=0)
            var = ((X - mean) ** 2).mean(axis=0)
        else:
            #卷积层
            mean = X.mean(axis=(0, 2, 3),keepdims=True)
            var = ((X - mean) ** 2).mean(axis=(0, 2, 3))
        X_hat = (X - mean) / nd.sqrt(var + eps)
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
        return  Y, moving_mean, moving_var

#定义BatchNorm层
class BatchNorm(nn.Block):
    def __init__(self, num_features, num_dims, **kwargs):
         super(BatchNorm, self).__init__(**kwargs)
         if num_dims ==2:
             shape = (1, num_features)
        else:
            shape = (1, num_feztures, 1, 1)      #卷积层各层独自归一化
        self.gamma = self.params.get('gamma',shape=shape, init=init.One())
        self.beta = sel.params.get('beta', shape=shape, init=init.Zero())
        self.moving_mean = nd.zeros(shape)
        self.moving_var = nd.zeros(shape)

    def forward(self, X):
        if self.moving_mean.context != X.context:
            self.moving_mean = slef.moving_mean.copyto(X.context)
            self.var_mean = self.var_mean.copyto(X.context)
        Y, self.moving_mean, self.moving_var = batch_norm(X, self.gamma.data(),
         self.beta.data(), self.momentum,self.moving_mean, self.moving_var,
          eps=1e-5, momentum=0.9)
        return Y
#在Lenet里添加批量归一化层
net = nn.Sequential()
net.add(nn.Conv2D(6, kernel_size=5),
        BatchNorm(6, num_dims=4),
        nn.Activation('sgmoid'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Conv2D(16, kernel_size=5),
        BtachNorm(16, num_dims=4),
        nn.Activation(;sigmoid),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Dense(120),
        BatchNorm(120, num_dims=2),
        nn.Activation('sigmoid'),
        nn.Dense(84),
        BatchNorm(84, num_dims=2),
        nn.Activation('sigmoid'),
        nn.Dense(10))

#训练模型
lr, num_epochs, batch_szie, ctx = 1.0, 5, 256, d2l.try_gpu()
net.initialize(ctx=ctx, init=init.Xivaer())
trainer = gluon.Trainer(net.collect.params(), 'sgd',{'learning_rate':lr})
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch5(net, teainer_iter, test_iter, batch_size, trainer, ctx, num_epochs)

#用自带的BatchNorm简洁实现
net = nn.Sequential()
net.add(nn.Conv2D(6, kernel_size=5),
        nn.BtchNorm(),
        nn.Activation('sigmoid'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Conv2D(16, kernel_size=5),
        nn.BatchNorm(),
        nn.Activation('sigmoid'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Dense(120),
        nn.BatchNorm(),
        nn.Activation('sigmoid'),
        nn.Dense(84),
        nn.BatchNorm(),
        nn.Activation('sigmoid'),
        nn.Dense(10))

net.initialize(ctx=ctx, init=init.Xavier())
trainer = gluon.Trainer(net.collect.parms(),'sgd',{'learning_rate':lr})
d2l.train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx, num_epochs)
