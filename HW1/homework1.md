# Homework 1
## 自己写的code
1. + import 包
   + class args: 定义超参数，写数据集(train.csv, test.csv)的path，logs的path
   
2. + load data： pandas read from
   
   + scaler, 用的MinMax, 用.fit_transform记录下所需的数据  
     q: 关于这里我有需要解答的问题，希望到时候看到Normalization那一节的时候能够解答: 在这里我们只scale了features，为什么不需要scale labels来达到统一？这里我对吴恩达的课没有什么印象。
     
     a: normalization是为了训练的时候更快达到最优解，和labels没有关系，同时这里我用的应该不算是batch normalization(又挖坑了)，直接把整个training data的数据用来计算了。
     
   + 用train_test_split split train into train & valid (random_state=args.seed)
   
3. + define Dataset: init(x, y, scaler(用.transform(features))), getitem, len
   + Dataset object --> Dataloader(dataset, batch_size, shuffle) --> test loader(这里可以得到input_dim和output_dim)
   
4. + Model: init, forward

5. + (记得老师的那个图) 先写出model(to device), loss_f, optimizer
   + tqdm
   + 开 train! 
     train_r2, valid_r2
     + 每个epoch: 每次都要在list中间加入 train_r2, valid_r2, 最后画这个图
       epoch_loss, epoch_loss_v, epoch_r2, epoch_r2_v, index_train, index_valid
       + 每个train的batch：epoch_loss, epoch_r2, index_train
         + data to device
         + 每次：
           ```python
           optimizer.zero_grad()
           pred = model(data)
           loss = loss_f(pred, label)
           loss.backward()
           optimizer.step()
         + 计算epoch_loss和epoch_r2, 每次都+=, 这样最后得到的值就是一个batch完结之后的
           r2_score(label.cpu().detach().numpy(),pre.cpu().detach().numpy()) 算r2的时候label在前，loss则反过来
         + index_train += 1并且显示par，说实话这个par里面的显示我觉得没必要弄得完全清楚
           + .set_description_str： epoch, iteration
           + .set_postfix_str: R2, loss, MAPE(感觉不是很重要，略过)
       + model.eval()
         with torch.no_grad():
         + 每个valid的batch：epoch_loss_v, epoch_r2_v, index_valid
           + data to device
           + 每次：
               ```python
               pred = model(data)
               loss = loss_f(pred, label)
           + 计算epoch_loss_v, epoch_r2_v, 每次都+=
           + par
       + save checkpoint: torch.save(model.state_dict(), f'{args.log_path}/Linear-{i+1}.pth') 相当于每个epoch都把模型存在logs里
       + train和valid的r2存入lists
       + np.savetxt把train_r2, valid_r2的两个lists存下来 --> 为了方便画图
   
6. 画图: 调用两个plot画在一张图上

7. 把valid_loader的batch_size改成1，load model，用pred_list=[]，label_list=[]存下来每个

8. 画图：画一条x=y的线，再画x=label y=pred的散点图，看

## HungyiLee-ML-HW1
+ 再次看老师的code的时候，我发现我犯了一个错误，我没有好好看数据，所以现在我们再来一起解析一下这个题
+ 题目描述：Given survey results in the past 5 days in a specific state in U.S., then predict the percentage of new tested positive cases in the 5th day.  
  意思是给我前四天的survey和positive cases，given第5天的survey，算第5天的positive cases
+ Data分析(这就是我错的点，我误以为除了最后一列之外其他的都是features, 并且第一列还是id)：
  但是看老师的code，其实好像对整个model没有任何影响，还是把除了最后一列之外的连id都算成了features(小丑还是我自己)
  + id
  + States (37, encoded to one-hot vectors)
    + One-hot vectors: Vectors with only one element equals to one while others are zero. Usually used to encode discrete values.
  + 重复5次：
    + COVID-like illness (4)
    + Behavior Indicators (8)
    + Mental Health Indicators (3)
    + Tested Positive Cases (1): This is what we want to predict.
      Test set没有第5天的(需要预测)
      我们可以借用我们之前的得到的train_loader里面的"features(误)"数量算一下：118 = 1 + 37 + (4 + 8 + 3 + 1) * 5
+ Hints 
  + simple : sample code
  + medium : Feature selection
  + strong : Different model architectures and optimizers
  + boss : L2 regularization and try more parameters： L2 regularization 除了 sample code 提供的在計算 loss 時處理之外，也可以使用 optimizer 的 weight_decay 實現，可參考 PyTorch
  官方文檔
+ 想到的问题：one-hot怎么运用到regression里，还是一样的结构吗 --> 是的
### Simple Baseline
1. 一样先import
2. Utility function: same_seed, train_valid_split, predict
3. 定义Dataset
4. 定义Model(修改到strong)
5. Feature Selection(这个是medium)
6. Training Loop: 这个是定义的一个函数
   def trainer(train_loader, valid_loader, model, config, device):
   + criterion, optimizer(different: strong, L2 regularization： boss)  
     tensorboard  
     创建models directory来save models  
     n_epochs, best_loss, step, early_stop_count = config['n_epochs'], math.inf, 0, 0 # math.inf 是无穷大，只要比这个小就会被更新
   + 每个epoch：
     + model.train()  
       loss_record = []  
       train_pbar = tqdm(train_loader, position=0, leave=True)
     + 每个train的batch：
       + .zero_grad  
         .to device  
         pred  
         loss  
         .backward  
         .step  
         step += 1
         loss_record.append  
         train_pbar: epoch 进度 loss
     + mean_train_loss  
       writer.add_scalar('Loss/train', mean_train_loss, step)
     + model.eval()  
       loss_record = [] # 感觉是因为覆盖了也没关系
     + 每个validation的batch:
       + .to device  
         .no grad:
         + pred  
           loss  
           loss_record.append
     + mean_valid_loss
       print mean_train_loss and mean_valid_loss  
       writer.add_scalar('Loss/valid', mean_valid_loss, step)
     + best_loss，存model，
7. device和config字典
8. Dataloader
   + same_seed
   + load data and split
     print
   + Select features
     print the number of features
   + dataset objects
   + dataloader objects
9. Start training!
   + model
   + train
10. tensorboard 画图
11. test的pred并保存为csv