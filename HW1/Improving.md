# Improving
这里我是根据老师的hints还有Github上的[参考1](https://github.com/Cangshanqingshi/Hung-Yi-Lee-Machine-Learning-Homework/tree/main/HW1_%E5%9B%9E%E5%BD%92_COVID_%E9%A2%84%E6%B5%8B),
[参考2](https://github.com/WSKH0929/LHY_DeepLearning_2022/tree/master/LHY_DeepLearning_2022/HomeWork01)来进行模型的改进。每个我选的是我相对好理解并且看懂了之后能够自主实现的。
## Medium
进行Feature Selection  
原代码：
```python
def select_feat(train_data, valid_data, test_data, select_all=True):
    '''Selects useful features to perform regression'''
    y_train, y_valid = train_data[:,-1], valid_data[:,-1]
    raw_x_train, raw_x_valid, raw_x_test = train_data[:,:-1], valid_data[:,:-1], test_data

    if select_all:
        feat_idx = list(range(raw_x_train.shape[1]))
    else:
        feat_idx = [0,1,2,3,4] # TODO: Select suitable feature columns.

    return raw_x_train[:,feat_idx], raw_x_valid[:,feat_idx], raw_x_test[:,feat_idx], y_train, y_valid
```
这是根据参考1的：
```python
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression
def choose_feature(train_data, valid_data, test_data, k=40, select_all=False):
    y_train, y_valid = train_data[:,-1], valid_data[:,-1]
    raw_x_train, raw_x_valid, raw_x_test = train_data[:,:-1], valid_data[:,:-1], test_data
    if select_all:
        feat_idx = list(range(raw_x_train.shape[1]))
    else:
        model = SelectKBest(f_regression, k=k)
        model.fit_transform(raw_x_train, y_train)
        scores = model.scores_
        indices = np.argsort(scores)[::-1]
        feat_idx = indices[0:k]
    
    return raw_x_train[:,feat_idx], raw_x_valid[:,feat_idx], raw_x_test[:,feat_idx], y_train, y_valid
```

## Strong
Different model architectures and optimizers  
model比较好改，改变激活函数和#layer nodes之类的  
首先是我自己写的：
```python
def __init__(self,input_dim=batch[0].shape[1],output_dim=batch[1].shape[1]):
    super(Net,self).__init__()
    self.model=nn.Sequential(
        # y=wx+b
        nn.Linear(input_dim,256,bias=True),
        nn.LeakyReLU(),
        nn.Linear(256,128,bias=True),
        nn.LeakyReLU(),
        nn.Linear(128,output_dim,bias=True),
    )
```
这是根据参考2的：
```python
self.layers = nn.Sequential(
        nn.Linear(input_dim, input_dim // 2),
        nn.LeakyReLU(),
        nn.Linear(input_dim // 2, 4),
        nn.LeakyReLU(),
        nn.Linear(4, 1),
    )
```
optimizer也有很多选择，我们初始用的是SGD。这里挖个坑，因为我看到参数里面有momentum，然后后面有的课应该会讲到。  
参考1： `optimizer = torch.optim.Rprop(model.parameters(), lr=0.000001)`  
参考2： Adam  
同时还要注意的是，"L2 regularization 除了 sample code 提供的在計算 loss 時處理之外，也可以使用 optimizer 的 weight_decay 實現"。  
example: `optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.001)`

## Boss
L2 正则化（L2 Regularization），也被称为 Ridge 回归，是一种用于防止机器学习模型过拟合（Overfitting）的技术。过拟合是指模型在训练数据上表
现很好，但在未见过的数据上表现不佳。L2 正则化通过向损失函数添加一个正则化项来限制模型的复杂度，从而帮助模型更好地*泛化*到新的数据。
λ * sum(param ** 2)
参考1：
```python
def get_loss(pred, target, model, loss_function):
        regularization_loss = 0
        for param in model.parameters():
            regularization_loss += torch.sum(param ** 2)
        return loss_function(pred, target) + 0.00075 * regularization_loss
```
参考2：这个就是只写了weight。通常只对模型的权重（weights）进行正则化，而不对偏置项（bias）进行正则化
```python
def cal_loss(self, pred, target):
        """ Calculate loss """
        # TODO: you may implement L1/L2 regularization here

        # Improve: L1 regularization
        # l1_lambda = 0.001
        # l1_loss = 0
        # for name, w in self.layers.named_parameters():
        #     if 'weight' in name:
        #         l1_loss += l1_lambda * torch.norm(w, p=1)
        #
        # return self.criterion(pred, target) + l1_loss

        # Improve: L2 regularization
        l2_lambda = 0.001
        l2_loss = 0
        for name, w in self.layers.named_parameters():
            if 'weight' in name:
                l2_loss += l2_lambda * torch.norm(w, p=2)

        return self.criterion(pred, target) + l2_loss
```
正则化参数 
λ 的选择对模型性能至关重要：
+ λ 太小：正则化效果不明显，模型可能过拟合。
+ λ 太大：正则化过强，模型可能欠拟合。
通常，通过交叉验证（Cross-Validation）来选择最优的 λ 值。