#训练模型main progrom
def train_ch3(net, train_iter, test_iter, loss ,num_epochs,batch_size,params=None,trainer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0,0.0,0
        for X, y in train_iter:
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat,y).sum()
            l.backward()
            if trainer is None:                                  #可以在函数之前自定义trainer
                d2l.sgd(params, lr, batch_size)
            else:
                trainer.step(batch_size)              #step函数会对参数迭代取平均值
            y = y.astype('float32')
            train_l_sum += l.asscalar()
            train_acc_sum += (y_hat.argmax(axis=1)==y).sum().asscalar()
            n += y.size
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f,test acc %.3f'
        % (epoch + 1,trian_l_sum / n, train_acc_sum /n,test_acc))
