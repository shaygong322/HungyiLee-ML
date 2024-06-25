# 01 Introduction of Deep Learning

## 1 Machine Learning ≈ Looking for Function



## 2 Different Types of Functions
+ Regression: outputs a scalar 
+ Classification: given options, outputs the correct one
+ Structured learning



## 3 Case Study

### 3.1 Function with Unknown Parameters
+ Model (based on domain knowledge): 

$$
y = b + wx_1
$$

+ x<sub>1</sub>: feature
+ w (weight), b (bias) are unknown parameters

### 3.2 Define Loss from Training Data
+ Loss is also a function, whose input are parameters(b, w) from model. 

  L(b, w): how good a set of values is. The bigger L is, the worse parameters are. 

  y_hat is the prediction while y (aka label) is the true value.
  
+  Probability distribution --> Cross-entropy

$$
e = |y - \hat{y}| \quad \text{MAE}
$$

$$
e = (y - \hat{y})^2 \quad \text{MSE}
$$

$$
L = \frac{1}{N} \sum(e)
$$

+ Error Surface --> contour map

### 3.3 Optimization

$$
w^\ast, b^\ast = \arg \min_{w, b} L
$$

Gradient Descent: 

+ (Randomly) Pick initial values w<sup>0</sup>, b<sup>0</sup>

+ Compute:

$$
\frac{\partial L}{\partial w} \bigg|_{w=w^0, b=b^0}
$$

$$
\frac{\partial L}{\partial b} \bigg|_{w=w^0, b=b^0}
$$

​	Can be done in one line in most deep learning frameworks

$$
w^1 \leftarrow w^0 - \eta \frac{\partial L}{\partial w} \bigg|_{w=w^0, b=b^0}
$$

$$
b^1 \leftarrow b^0 - \eta \frac{\partial L}{\partial b} \bigg|_{w=w^0, b=b^0}
$$

+ Update w and b iteratively



## 4 Linear Model

bias + weight * feature

$$
y = b + \sum_{j=1}^{7} w_j x_j
$$

$$
y = b + \sum_{j=1}^{28} w_j x_j
$$

$$
y = b + \sum_{j=1}^{56} w_j x_j
$$



## 5 Piecewise Linear Curves

+ Linear model has limitation: the bias of the model. 

+ constant + a set of hard sigmoid --> piecewise linear curves --> any continuous curve

+ sigmoid function --> hard sigmoid

$$
y = \frac{c}{1 + e^{-(b + wx_1)}} \\
= c \cdot \text{sigmoid}(b + wx_1)
$$

  w bigger slope steeper

  b bigger slope shift left

  different c change height 

+ New Model: 

$$
y = \frac{c}{1 + e^{-(b + wx_1)}} \\
= c \cdot \text{sigmoid}(b + wx_1)
$$

### More Features

$$
y = b + \sum_i c_i \cdot \text{sigmoid}\left(b_i + \sum_j w_{i,j} x_j\right)
$$

+ j stands for the index of features
+ w_ij: weight for x_j for the i_th sigmoid
+ θ: unknown parameter



## 6 Back to ML_Step2: define loss from training data

$$
L = \frac{1}{N} \sum(e)
$$



## 7 Back to ML_Step3: Optimization

$$
\theta^* = \arg \min_{\theta} L
$$

+ (Randomly) Pick initial values θ<sup>0</sup>

+ Compute gradient

$$
g = \nabla L(\theta^0)
$$

$$
\theta^1 \leftarrow \theta^0 - \eta g
$$

### Batch, Epoch and Update
+ Divide N into several batches, every batch computes its own L and g

$$
g = \nabla L^1(\theta^0)
$$

  update

$$
\theta^1 \leftarrow \theta^0 - \eta g
$$

+ 1 epoch = see all the batches once

+ every parameter renew is one update (after one batch)

  eg. N = 10000, Batch size B = 10, then 1000 updates in 1 epoch.

### Activation functions
2 ReLU --> hard sigmoid 

$$
y = b + \sum_{2i} c_i \max \left(0, b_i + \sum_j w_{ij} x_j \right)
$$

### Deeper Model