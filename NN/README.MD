## 四种分类方法比较（自编底层代码）

1、简单多分类感知机

![](http://owmmu20d1.bkt.clouddn.com/17-9-29/15854290.jpg)

准确率为：51.5 %

2、简单神经网络（单层，平方误差+简单梯度下降）

![](http://owmmu20d1.bkt.clouddn.com/17-9-29/75172782.jpg)

准确率为： 63.5 %

3、复杂神经网络（4层sigmoid（64个神经元）+最后sotfmax和cross_entropy损失函数）

![](http://owmmu20d1.bkt.clouddn.com/17-9-29/18809522.jpg)

准确率： 96.5 %

4、sklearn自带SVC方法

![](http://owmmu20d1.bkt.clouddn.com/17-9-29/13176873.jpg)

准确率： 93.5 %