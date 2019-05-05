import d2lzh as d2l
import mxnet as mx
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import loss as gloss, nn
import time
#搭建网络模型

net = Sequential()   #创建容器
net.add(nn.Conv2D(channels=6,kernel_size=5,activation='sigmoid')
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Conv2D(channels=16, kernels_size=5,activation='sigmoid')
        nn.MaxPool2D(pool_size=2, strides=2),    #定义两个卷积层加两个池化层
        nn.Dense(120, activation='sigmoid'),
        nn.Dense(84, activation='sigmoid'),
        nn.Dense(10))

#获取数据，d2l.load_data_fashion_mnist(batch_size=batch_size)
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)
#用Gpu计算，目前不会使用Linux，所以没有跑代码
def try_gpu():
    try:
        ctx = mx.gpu()
        _ =nd.zeros((1,),ctx=ctx)
        except mx.base.MXNetError:
            cx = mx.cpu()
        return ctx
ctx = try_gpu()
ctx

#在定义函数时，要将数据复制到gpu的显存上来 ，as_in_context函数
def evaluate_accuracy(data_iter, net, ctx):
    acc_sum, n = nd.array([0], ctx=ctx), 0
    for X, y in data_iter:
        X,y = X.as_in_context(ctx), y.as_in_context(ctx).astype('float32')
        acc_sum += (net(X).argmax(axis=1)==y).sum()    #softmax原理
        n += y.size
    return acc_sum.asscalar() / n

def train_ch5(net, train_iter, test_iter, batch_size,trainer,ctx,num_epochs):
    print('training on', ctx)
    loss = gloss.SoftmaxCrossEntropyLoss()
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X,y in train_iter:
            X, y = X.as_in_context(ctx), y.as_in_context(ctx)
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y).sum()
            l.backward()
            trainer.step(batch_size)
            y = y.astype('float32')
            train_l_sum += l.asscalar()
            train_acc_sum += (y_hat.argmax(axis=1)==y).mean().asscalar()
            n += y.size
        test_acc = evaluate_accuracy(test_iter, net, ctx)
        print('epoch %d, loss %.4f,train acc %.3f,test acc %.3f, time %.1f sec '
                % (epoch + 1, train_l_sum / n,train_acc_sum / n, test_acc_sum, time.time() - start))

#训练
lr, num_epochs = 0.9, 5
net.initialize(force_reinit=True, ctx=ctx, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd',{'learning_rate':lr})
train_ch5(net, train_iter, test_iter, batch_size,trianer,ctx, num_epochs)
