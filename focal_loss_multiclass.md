# 用于多分类的focal-loss
## 原理
$focal-loss$ 是由交叉熵损失函数改进得来的。对于样本i，类别c而言，多分类的交叉熵公式为： $CE(i,c)=-{y_ic*log(p_ic)}$

其中 $p_ic$ 指的是样本i属于类别c的概率；当样本 $i$ 的真实类别为 $c$ 时 $y_ic=1$ , 否则 $y_ic=0$ 。  

在此基础上：  
* 引入每一个类别所占的权重 $\alpha$ -> 解决不同类别样本的数量不平衡问题
* 引入调节因子 $(1-p)^\gamma$ -> 解决难易样本的学习问题
  
则对于样本 $i$，类别 $c$ 而言， $focal-loss$ 的形式为：  $FL(i,c)=-\alpha _c*(1-y_ic * p_ic)^\gamma*log(y_ic * p_ic)$  

为什么要这样设置，可以这样理解：  

(1)通过引入权重，我们可以让数量较少的一类样本拥有更高的权重，让这部分样本的 $loss$ 占据更加重要的地位，从而让模型给予这些数量较少的样本更多的关注。  

(2)调节因子 $(1-p)^\gamma$ 实际上衡量了“预测值”和“真实值”之间的距离。如果 $p$ 趋近于1，表明该样本的预测值和真实值十分接近，是容易学习的样本，那么就在其 $loss$ 上乘以一个调节因子 $(1-p)^\gamma$ 使其变得更小，让模型减少对易分样本的关注。
## 参数
* 待预测样本 $input$ 是一个形如 $(N C H W)$ 的 $tensor$ ，元素值为每个像素属于各类别的概率值, 如0.56，0.39. 
* 真实标签 $target$ 是一个形如 $(N H W)$ 的 $tensor$ , 元素值为每个像素所属类别的索引值, 如0，1，2.
* 调节因子 $\gamma$ 是取值范围为[0,5]的整数, 一般取2.
* 权重参数 $\alpha$ 是一个 $list$ , 存放各个类别的权重，比如[1,1,1].
* 类别数 $classnum$ 是一个整数。
* 是否计算平均值 $avg$ 是一个布尔型变量。
  
## 说明
1. 将输入的 $input$ 、 $target$ 做的维度变换如下：   
  * $input:(N,C,H,W)-$ `.permute(0,2,3,1)` $->(N,H,W,C)-$ `.flatten(0,2)`- $>(NHW,C)$  
  * $target:(N,H,W)-$ `one_hot(dim=C)` $->(N,H,W,C)-$ `.permute(0,2)`- $>(NHW,C)$  

2. 利用掩膜运算，计算 $y_ic * p_ic$ 及 $log(y_ic * p_ic)$ ：  
  * 以target为参考，target=0所对应位置处的input概率设置为0，可以计算得到 $y_ic * p_ic$
  * 以target为参考，target=0所对应位置处的input概率设置为1，再取对数，可以计算得到 $log(y_ic * p_ic)$
  
## 参考  
* https://github.com/clcarwin/focal_loss_pytorch 
* https://www.jianshu.com/p/437ce8ed0413
* https://www.zhihu.com/tardis/zm/art/35709485?source_id=1005
* https://zhuanlan.zhihu.com/p/492396800
