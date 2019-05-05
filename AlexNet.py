import d2lzh as d2l
from mxnet import gluon, init ,nd
from mxnet.gluon import data as gdata, nn
import os
import sys
#搭建模型
net = nn.Sequential()
net.add(nn.Conv2D(96, kernel_size=11, strides=4, activcation='relu'),
        nn.MaxPool2D(pool_size=3, strides=2),
        nn.Conv2D(256, kernel_size=5,padding=2,activation='relu'),
        nn.MaxPool2D(pool_size=3, strides = 2),
        nn.Conv2D(384, kernel_size=3, padding=1, activation='relu'),
        nn.Conv2D(384, kernel_size=3, padding=1, activation='relu'),
        nn.Conv2D(256, kernel_size=3, padding=1, activation='relu'),
        nn.MaxPool2D(pool-size=3, strides=2),
        nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
        nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
        nn.Dense(10))
#查看每层的输出
X = nd.random.uniform(shape=(1, 1, 224, 224))
net.initialize()
for layer in net:
    X = layer(X)
    print(layer.name, 'output shape:\t',X.shape)
#下载并处理数据
def load_data_fashion_mnist(batch_size, resize=None, root=os.path.join('~','mxnet', 'datasets', 'fashion-mnist')):
    root = os.path.expanduser(root)
    transformer= []         #为transformer创建列表
    if resize:
        transformer += [gdata.vision.tranforms.Resize(resize)]     #改变图像形状
    transformer += [gdata.vision.transforms.ToTensor()]             #将图像的最后一维通道至于第一维，且将二维像素值转为float32再除以255
    transformer = gdata.vision.transforms.Compose(transformer)      #连接两步处理，Resize要在ToTensor之前
    mnist_train = gdata.vision.FashionMNIST(root=root, train=True)
    mnist_test = gdata.vision.FashionMNIST(root=root,train=False)
    num_workers = 0 if sys.platform_startwith('win32') else 4             #非Windows系统可以使用多进程读取数据
    train_iter = gdata.DataLoader(
        mnist_train.transform_first(transformer), batch_size, shuffle= True,          #transform_first(transformer)对第一个元素做处理
        num_workers = num_workers)
    test-iter = gdata.DataLoader(
        mnist_test.transform_first(transformer), batch-size, shuffle= False,
        num_workers = num_workers)
    return train_iter, test_iter

batch_size = 128
train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=224)

lr,num_epochs, ctx = 0.01, 5 , d2l.try_gpu()
net.initialize(force_reinit=True, ctx=ctx, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(),'sgd', {'learning_rate: lr'})
d2l.train_ch5(net, train_iter, test_iter, batch_size, train, ctx ,num_epochs)