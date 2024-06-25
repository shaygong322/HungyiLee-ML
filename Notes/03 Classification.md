# 03 Classification

## 1 Classification as Regression

+ 回归是输入一个向量 x，输出 y_hat，我们希望 y_hat 跟某一个标签 y 越接近越好
+ 分类可当作回归来看，输入 x 后，输出仍然是一个标量 y_hat，我们可以把类也变成数字。类1是编号1，类2是编号2，类3是编号3，要让 y_hat 跟类的编号越接近越好
    + 但该方法在某些状况下会有问题，用数字来表示类会预设 1 和 2 有比较近的关系，1 和 3 有比较远的关系。但假设三个类本身没有特定的关系，分别独立。

这种情况，需要引入 one-hot vector (独热向量)来表示类。实际上在做分类的问题的时候，比较常见的做法也是用 one-hot vector 表示类。

### Class as one-hot vector

$$
\hat{y} = 
\begin{cases}
\begin{bmatrix}
1 \\
0 \\
0
\end{bmatrix} & \text{Class 1} \\
\begin{bmatrix}
0 \\
1 \\
0
\end{bmatrix} & \text{Class 2} \\
\begin{bmatrix}
0 \\
0 \\
1
\end{bmatrix} & \text{Class 3}
\end{cases}
$$

如果目标 $y$ 是一个有三个元素的向量，网络也要输出三个数字才行。输出三个数值就是把本来输出一个数值的方法，重复三次。把 $a_1, a_2$ 和 $a_3$ (比如 wx + b 再 Sigmoid 之后的数)乘上三个不同的权重，加上偏置，得到 $\hat{y}_1$；再把 $a_1, a_2$ 和 $a_3$ 乘上另外三个权重，再加上另外一个偏置得到 $\hat{y}_2$；把 $a_1, a_2$ 和 $a_3$ 再乘上另外一组权重，再加上另外一个偏置得到 $\hat{y}_3$。输入一个特征向量，产生 $\hat{y}_1, \hat{y}_2, \hat{y}_3$，希望 $\hat{y}_1, \hat{y}_2, \hat{y}_3$ 跟目标越接近越好。



## 2 Classification with Softmax

+ 按照上述的设定, Classification 实际过程是：输入 $x$, 乘上 $W$ 加上 $b$, 通过激活函数 $\sigma$, 再乘上 $W'$ 加上 $b'$ , 得到向量 $\hat{y}$. 
+ 但实际做分类的时候，往往会把 $\hat{y}$ 通过 softmax 函数得到 $y'$，才去计算 $y'$ 跟 $\hat{y}$ 之间的距离。

`Q：为什么分类过程中要加上 softmax 函数?`

` A：一个比较简单的解释是，y 是独热向量，所以其里面的值只有 0 跟 1，但是 y_hat 里面有任何值。既然目标只有 0 跟 1，但 y_hat 有任何值，可以先把它归一化到 0 到 1 之间，这样才能跟标签的计算相似度。`

### Softmax

$$
y'_i = \frac{\exp(y_i)}{\sum_j \exp(y_j)}\\
\begin{cases}
1 > y'_i > 0 \\
\sum_i y'_i = 1
\end{cases}
$$

+ Softmax 除了 normalized 让 y₁' y₂' y₃' 变成 0 到 1 之间, 还有和为 1 以外, 它还有一个附带的效果是, 它会让大的值跟小的值的差距更大
+ 两个类也可以直接套 softmax 函数。但一般有两个类的时 候，我们不套 softmax，而是直接取 sigmoid。当只有两个类的时候，sigmoid 和 softmax 是 等价的。



## 3 Loss of Classification
+ Mean Square Error

$$
e = \sum_i (\hat{y}_i - y'_i)^2
$$

+ Cross-entropy

$$
e = - \sum_i \hat{y}_i \ln y'_i
$$

Minimizing cross-entropy is equivalent to maximizing likelihood.

+ 在优化的时候，期望从损失大到损失小。均方误差在损失很大的地方非常“平坦”，其梯度非常小，趋近于0。如果初始时在损失大的位置，离目标非常远，梯度又很小，无法用梯度下降顺利地“走”到损失小的地方。