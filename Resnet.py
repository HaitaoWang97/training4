import d2lzh as d2l
from mxnet import gluon, init, nd
from mxnet.gluon import nn
#定义残差块，两个保持输入不变的卷积层并且每个卷积层后都加了BatchNorm加速网络计算
class Residual(nn.Block):
    def __init__(self, num_channels, use_1×1conv=False, strides=1, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.conv1 = nn.Conv2D(num_channels, kernel_size=3, padding=1, strides=strides)
        self.conv2 = nn.Conv2D(num_channels, kernel_size=3, padding=1)
        if use_1×1conv:           #当卷积层的通道数与输入不同时，需要对输入做一个1×1的conv改变其通道后再将两条线路相加
            self.conv3 = nn.Conv2D(num_channels, kernel_size=1, strides = strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm()  #批量归一化层
        self.bn2 = nn.BatchNorm()

 #定义前向计算过程   
    def forward(self, X):
        Y = nd.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if slef.conv3:
            X = self.conv3(X)
        return nd.relu(Y + X)
#Resnet的第一个模块与goolenet相同，添加了BatchNorm层
net = nn.Sequential()
net.add(nn.Conv2D(64, kernel_szie=7, strides=2, padding=3),
        nn.BatchNorm(), Activation('relu'),
        nn.MaxPool2D(pool_szie=3, strides=2, padding=1))
#定义残差层，第一个残差块的输入与输出i相同，其余残差块通道数比上一个模块翻倍，高宽减半
def resnet_block(num_channels, num_residuals, first_block=False):
    blk = nn.Sequential()
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.add(Residual(num_channels, use_1×1conv=True, strides= 2))
        else:
            blk.add(Residuals(num_channels))
    return blk
#定义四个残差层，每个残差层两个残差块
net.add(resnet_block(64, 2, first_block=True),
        resnet_block(128, 2),
        resnet_block(256, 2),
        resnet_block(512, 2))
#最后使用全局平均池化层和全连接层输出
net.add(nn.GlobalAvgPool2D(), nn.Dense(10))
#初始化模型，检查每层的输出形状
X = nd.random.uniform(shape=(1, 1, 224, 224))
ney.initialize()
for layer in net:
    X = layer(X)
    print(layer.name, 'output shape:\t', X.shape)
#训练模型
lr, num_epochs, batch_size, ctx = 0.05, 5, 256, d2l.try_gpu
net.initialize(force_reinit=True, ctx=ctx, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate':lr})
train_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
d2l.train_ch5(net, train_iter, test_iter, batch_size,trainer, ctx, num_epochs)

