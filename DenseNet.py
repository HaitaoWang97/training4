import d2lzh as d2l
from mxnet import gluon, init, nd
from mxnet.gluon import nn
#DensNet将ResNet的结构改成了“BatchNorm,activation,conv”
def conv_block(num_channels):
    blk = nn.Sequential()
    blk.add(nn.BatchNorm(), Activation('relu'), nn.Conv2D(num_channels, kernel_size=3,padding=1))
    return blk

#定义稠密块，由多个conv_block组成
class DenseBlock(nn.Block):
    def __init__(self, num_convs, num_channels, **kwargs):
        super(DenseBlock, self).__init__(**kwargs)
        self.net = nn.Sequential()
        for _ in range(num_convs):
            self.net.add(conv_block(num_channels))
#将Denseblock的各conv_block输出与输入按通道连接
    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            X = nd.concat(X, Y, dim=1)
        return X

#以两个10通道数的conv_block组成的Denseblock为例
blk = DenseBlock(2, 10)
blk.initialize()
X = nd.random.uniform(shape=(4, 3, 8, 8))
Y = blk(X)
Y.shape
#会输出(4, 23, 8, 8)

#用过渡层控制连接的通道数过大
def transition_block(num_channels):
    blk = nn.Sequential()
    blk.add(nn.BatchNorm(), nn.Activation('relu'),
            nn.Conv2D(num_channels, kernel_size=1),
            nn.AvgPool2D(pool_szie=2, strides=2))
    return blk

blk = transition_block(10)
blk.initialize()
blk(Y).shape

#搭建模型
net = nn.Sequential()
net.add(nn.Conv2D(64, kernel_size=7, strides=2, padding=3),
        nn.BatchNorm(), nn.Activation('relu'),
        nn.MaxPool2D(pool_size=3, strides=2, padding=1))

#添加四个Denseblock
num_channels, growth_rate = 64, 32 #当前通道， conv_block通道
num_convs_in_dense_blocks = [4, 4, 4, 4]

for i, num_convs in enumerate(num_convs_in_dense_block):    #enumerate同时列出索引值和数据值
    add(DenseBlock(num_convs, growth_rate))
    num_channels += num_convs * growth_rate
    if i != len(num_convs_in_dense_blocks) - 1:
        num_channels //= 2
        net.add(transition_block(num_channels))

net.add(nn.BatchNorm(), nn.Activation('relu'), nn.GlobalAvgPool2D(), nn.Dense(10))

lr, num_epochs, batch_size, ctx = 0.1, 5, 256, d2l.try_gpu()
net.initailize(ctx=ctx, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd',{'learning_rate':lr})
trainer_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx, num_epochs)
