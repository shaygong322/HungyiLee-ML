# 02 What to do if my network fails to train

这一节介绍一个基本的训练过程里有哪些需要注意和可以改进的地方

## 1. General Guidance: 基本的框架和怎样更好的训练

### 1.1 Framework of ML

三步训练：1 写出含有未知参数的function 2 写出关于未知参数的function -- loss function 3 Optimization

sample code --> simple baseline, but how should we improve it?

### 1.2 General Guide

如果模型预测结果不好，先去检查training loss，再看testing结果，如果training loss很大，训练集训练的不好，那么：

#### Training结果不好，可能出现的两个问题：

#### 1.2.1 Model Bias

或者用通俗的话来说，模型太过简单，function set里甚至不含有能让loss足够低的函数

解决方法：1 增加features 2 **deep** learning

#### 1.2.2 Optimization Issue

陷入local minima，虽然想找的function在set里，却动弹不得，无法找出

#### 两者怎么区分：

先跑一些shallower networks(选择比较容易optimize的，避免优化失败)，如果在这个比较小比较浅的network增加层数和复杂度之后的model，明明弹性更大更复杂，loss却没有更低，这就是optimization issue。

假设training loss已经变小了，但是testing data loss大，就真的可能是overfitting：

#### Training的loss小，testing的loss大：

#### 1.2.3 Overfitting

解决办法：

1 增加训练集：可以使用data augmentation来扩充数据

2 不要让模型太flexible: 1 less parameters (CNN) 2 less features 3 Early stopping 4 Regularization 5 Dropout (部分神经元禁用)

#### Bias-Complexity Trade-off: 

#### 1.2.4 Cross Validation

training --> training + validation

如何合理地区分training set和validation set --> N-fold validation: 把训练集切成N等份，拿其中一个当作Validation set，重复N次，把N个model在相同环境下train和valid都跑一次，每个model这N种数据集的结果都平均起来，看谁的最好

#### 1.2.5 Mismatch

训练集和测试集的分布不一样，这样的实验没有意义



## 2 Optimization

### 2.1 Critical point

When gradient is small，除了有local minima，还有saddle point

#### 2.1.1 如何判断是local minima还是saddle point

θ附近Loss泰勒展开 --> Hessian Matrix H, 只需考察H的特征值：

1 eigen values全正，H possitive definite 正定矩阵 --> local minima

2 全负，local maxima

3 有正有负 --> saddle point

#### 2.1.2 Don't be afraid of saddle point

θ <-- 负的eigen value对应的eigen vector + θ' (运算量大，实际中没人用这个)

#### local minima没有那么常见，gardient很小不再update很多情况下是saddle point

### 解决卡在critical point的办法：

### 2.2 Batch

#### 2.2.1 Shuffle -- 每一个Epoch的Batch都不一样

#### 2.2.2 Small Batch vs Large Batch

1 因为GPU并行计算，大批次(不是太大)和小批次单次时间差不多，但是1个Epoch大批次耗时更短

2 小批次performance更好

3 Noisy update有助于训练，因为每次更新loss函数都有差异，不同batchbutongloss function，更不容易卡住

4 Noisy update有利于testing，因为更不容易overfitting；还有更容易走入“平原”的minima、

### 2.3 Monentum

加上一个惯性，考虑所有的gradients

### 2.4 Adaptive Learning Rate

#### 2.4.1 Training stuck不代表small gradient

而有可能是振荡 --> learning rate设定太大

同时采用固定的lr很难到达最优解：1 lr较大 --> 最优解附近来回横跳 2 lr较小，最开始朝着最优解稳定移动，靠近后移动缓慢

所以，没有哪个learning rate能一劳永逸适应所有的参数更新速度

#### 2.4.2 Different parameters need different learning rates

$$
\theta_i^{t+1} \leftarrow \theta_i^t - \frac{\eta}{\sigma_i^t} g_i^t
$$

**基本原则：**

+ 某一个方向上gradient的值很小,非常的平坦 ⇒ learning rate调大一点

+ 某一个方向上非常的陡峭,坡度很大 ⇒ learning rate可以设小一点

##### 1 Root mean square --> 用于Adagrad

$$
\sigma_i^t = \sqrt{\frac{1}{t+1} \sum_{i=0}^{t} \left(g_i^t\right)^2}
$$

缺点：累积 --> 不能 “实时” 考虑梯度的变化情况

##### 2 RMSprop

$$
\sigma_i^t = \sqrt{\alpha (\sigma_i^{t-1})^2 + (1 - \alpha) (g_i^t)^2}
$$

添加参数 α（表示当前梯度大小对于 learning rate 的影响比重，是一个超参数（hyperparameter)

- α 设很小趋近於0，就代表这一步算出的 gᵢ 相较于之前所算出来的 gradient 而言比较重要

- α 设很大趋近於1，就代表现在算出来的 gᵢ 比较不重要，之前算出来的 gradient 比较重要

##### 3 Adam = RMSprop + Momentum 最常用的optimizer

其中使用 Pytorch 预设的参数就会有很好的效果

#### 2.4.3 Learning rate scheduling

训练到后面时，由于梯度的积累，会导致整体的 learning rate 极速增加，从而发生抖动，但随着新的大梯度的加入，会使得逐渐learning rate降低，最终梯度逐渐平稳。

**解决方法**：Learning Rate Scheduling ⇒ 让 LearningRate 与 “训练时间” 有关。将分子 也进行调整，将其升级为与时间相关的一个变量 （使用 Warm Up的方式，随着时间先变大后变小）。

$$
\theta_i^{t+1} \leftarrow \theta_i^t - \frac{\eta^t}{\sigma_i^t} g_i^t
$$

##### Learning rate decay

- 这种策略用于在训练的过程中逐渐减少学习率。随着训练接近目标，我们减少学习率以防止跳过最优解。这有助于在优化的后期更精细地调整参数，达到更好的优化效果。

##### Warm up

+ 这种策略在训练初期先增大学习率，然后再减小。在训练初期，由于估计的 σ具有较大的方差，所以采用较高的学习率可以快速收敛到较好的区域。之后再逐渐减小学习率，以细化对参数的调整。
+ 在训练神经网络或其他机器学习模型时，较大的方差可能意味着初始阶段的梯度估计不够准确，参数更新方向可能会有较大的波动。为了在这种情况下更快地找到一个较好的参数区域，可以使用较大的学习率进行快速探索。这就是预热策略的核心思想。

在训练初期，由于对误差表面状态的估计还不够精确，使用较小的学习率可以避免参数走得太远，从而防止模型训练时出现不稳定的情况。随着训练的进行，估计逐渐变得准确，可以适当提高学习率，以更有效地优化参数。

### 2.5 Optimization Summary

$$
\theta_i^{t+1} \leftarrow \theta_i^t - \frac{\eta^t}{\sigma_i^t} m_i^t
$$



## 3 Batch Normalization

归一化的目的主要是为了让模型的收敛速度更快，对于使用梯度下降优化的模型，每次迭代会找到梯度最大的方向迭代更新模型参数。但是，如果模型的特征属性量纲不一，那么寻求最优解的特征空间，就可以看做是一个椭圆形的，其中大量冈的属性对应的参数有较长的轴。在更新过程中，可能会出现更新过程不是一直朝向极小点更新的，而是呈现Z字型。使用了归一化对齐量纲之后，更新过程就变成了在近似圆形空间，不断向圆心（极值点）迭代的过程。

### 3.1 Feature normalization/scaling

mean是一行数据算出来的

$$
x_i^r \leftarrow \frac{x_i^r - m_i}{\sigma_i}
$$

In general, feature normalization makes gradient descent converge faster.

### 3.2 Considering Deep learning

normalization 可以apply在activation function 的input/output，但现在比较多的是**对activation function的input做normalization**

z -- sigmoid --> a

一个batch算mean和divation

$$
\tilde{z}^i = \frac{z^i - \mu}{\sigma}
$$

有时候，你并不希望你得activation function input的 mean=0， standard divation = 1，所以你可以做以下操作，同时也会跟随网络更新:

$$
\hat{z}^i = \gamma \odot \tilde{z}^i + \beta
$$

mean和standard divation受data影响，β和γ是network学出来的

### 3.3 Testing

We do not always have batch at testing stage.

Computing the moving average of 𝝁 and 𝝈 of the batches during training.

$$
\bar{\mu} \leftarrow p \bar{\mu} + (1 - p) \mu^t
$$

$$
\tilde{z} = \frac{z - \bar{\mu}}{\bar{\sigma}}
$$

### 3.4 Internal Covariate Shift

对每一个layer做feature scaling对Deep learning上是由很大作用的，他会让internal covariate shift（内部协方差平移）问题轻微一些，因为使得每层输入分布稳定。（未完待续）