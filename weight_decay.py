
%matplotlib inline
import dlzh as d2l
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, nn
#生成样本数据集
n_train, n_test, num_inputs = 20, 100, 200
true_w, true_b = nd.ones((num_inputs, 1)) * 0.01 ,0.05

features = nd.random.normal(shape=(n_train + n_test, num_inputs))
labels = nd.dot(features, true_w) + true_b
labels += nd.random.normal(scale=0.01, shape=labels.shape)
train_features, test_features = features[:n_train, :],features[n_train:, :]
train_labels, train_labels = labels[:n_train],labels[n_train:]
#初始化模型
def init_params():
    w = nd.random.normal(scale=1,shape=(num_inputs,1))
    b = nd.zeros(shape=(1,))
    w.attach_grad()
    b.attach_grad()
    return [w, b]
#定义L2范数惩罚项
def l2_penalty(w):
    return (w**2).sum() / 2
#定义训练
batch_size, num_epochs, lr = 1, 100 , 0.003
net, loss = d2l.linreg, d21.squared_loss
train_iter = gdata.DataLoader(gdata.ArrayDataset(train_features, train_labels),batch_size,shuffle=True)

def fit_and_plot(lambd):
    w, b = init_params()
    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for X,y in train_iter:
            with autograd.record():
                l = loss(net(X, w, b), y) + lambd * l2_penalty(w)
            l.backward()
            d2l.sgd([w, b], lr, batch_size)
        train_ls.append(loss(net(train_features, w, b),
        train_labels).mean().asscalar)
        test_ls.append(loss(net(train_features, w, b),
        test_labels).mean().asscalar)
    d2l.semilogy(range(1, num_epochs + 1),train_ls,'epochs','loss',
                range(1, num_epochs +1),trest_ls,['train','test'])

#未加入L2惩罚项
fit_and_plot(lambd=0)
#惩罚项超参数为3
fit_and_plot(lambd=3)

#直接在Trainer实例通过wd参数指定权重衰减参数
#简洁实现
def fit_and_plot_gluon(wd):
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize(init.Normal(sigma=1))
    trainer_w = gluon.Trainer(net.collect_params('.*weight'),'sgd',{'learning_rate':lr})
    trainer_b = gluon.Trainer(net.collect_params('.*bias'),'sgd',{'learning_rate':lr})
    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for X, y in train_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer_w.step(batch_size)
            trainer_b.step(batch_size)
        train_ls.append(loss(net(train_features),train_labels).mean().asscalar())
        test_ls.append(loss(net(test_features),test_labels).mean().asscalar())
    d2l.semilogy(range(1, num_epochs+1),train_ls,'epochs','loss',
                range(1,num_epochs+1), test_ls,['train','test')
    print('L2 norm of w:', net[0].weight.data().norm().asscalar())
#未加衰减
fit_and_plot_gluon(0)
#加衰减
fit_and_plot_gluon(3)