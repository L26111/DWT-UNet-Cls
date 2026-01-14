import os
import random
import cv2
import paddle
import numpy as np
from paddle.io import Dataset
from paddle.vision import transforms
from MydatasetCxWDB import CustomDataset
from MyAccuracy import Accuracy
import time
import csv  

# 训练记录数据使用的列表
# train_step_losses = []      #batch粒度的损失
# train_step_acc = []      #batch粒度的acc
# train_epoch_losses = []      #epoch粒度的损失
# train_epoch_acc = []      #epoch粒度的acc
# 验证记录信息的变量定义
# 初始化全局变量为None，避免累计旧数据
y_pred = None
y_true = None
y_outputForTsne = None
y_down2 = None
y_down4 = None
y_mid_bn2 = None
y_up3 = None
y_up1 = None
# dev_losses=[]
# dev_scores=[]

def evaluate(model, metric, dev_loader, Ispredict, epoch):
    # 声明使用全局变量
    global y_pred, y_true, y_outputForTsne, y_down2, y_down4, y_mid_bn2, y_up3, y_up1
    # 重置全局变量，避免跨epoch累计
    y_pred = []
    y_true = []
    y_outputForTsne = None
    y_down2 = None
    y_down4 = None
    y_mid_bn2 = None
    y_up3 = None
    y_up1 = None

    # 将模型设置为评估模式
    model.eval()
    # 关闭梯度计算，节省内存并加速
    with paddle.no_grad():
        # 用于统计训练集的损失
        total_loss = 0

        # 重置评价指标（确保每个epoch重新计算）
        metric.reset() 
        
        # 遍历验证集每个批次    
        for batch_id, data in enumerate(dev_loader):
            X, y = data
            # 计算模型输出
            logits = model(X)

            # 记录混淆矩阵数据
            if Ispredict:
                y_pred.append(logits)
                y_true.append(y)
            
            # 计算损失函数
            loss_fn = paddle.nn.CrossEntropyLoss()
            loss = loss_fn(logits, y).item()
            # 累积损失
            total_loss += loss 

            # 累积评价
            metric.update(logits, y)

            # 记录tsne数据（使用模型中保存的fc1输出特征）
            current_fc1 = model.y_fc1  # 从模型获取中间特征
            if batch_id == 0:
                y_outputForTsne = current_fc1.numpy()
            else:
                y_outputForTsne = np.vstack((y_outputForTsne, current_fc1.numpy()))
            
            # 记录WaveletAttention相关特征（从模型获取中间层输出）
            # 累积所有批次的特征（或根据需求调整）
            if batch_id == 0:
                y_down2 = model.down2_out.numpy()
                y_down4 = model.down4_out.numpy()
                # 若模型中有mid_bn2、up3、up1的输出保存，此处对应获取
                # y_mid_bn2 = model.mid_bn2_out.numpy()
                # y_up3 = model.up3_out.numpy()
                # y_up1 = model.up1_out.numpy()
            else:
                y_down2 = np.vstack((y_down2, model.down2_out.numpy()))
                y_down4 = np.vstack((y_down4, model.down4_out.numpy()))
                # y_mid_bn2 = np.vstack((y_mid_bn2, model.mid_bn2_out.numpy()))
                # y_up3 = np.vstack((y_up3, model.up3_out.numpy()))
                # y_up1 = np.vstack((y_up1, model.up1_out.numpy()))

            print(f"IN eval, batch id {batch_id} is completed!")
        
        # 存储tsne数据文件
        import pickle
        if Ispredict:
            with open('/home/aistudio/logsCSP_x_NewWave/T-OutputForTsne.pkl', 'wb') as f:
                pickle.dump(y_outputForTsne, f)
            
            # 处理预测结果
            y_pred1 = paddle.concat(y_pred, axis=0)
            y_pred2 = y_pred1.numpy()
            y_pred3 = np.argmax(y_pred2, axis=-1)
            y_pred_final = y_pred3.flatten()
            with open('/home/aistudio/logsCSP_x_NewWave/T-ypred.pkl', 'wb') as f:
                pickle.dump(y_pred_final, f)
            
            # 处理真实标签
            y_true1 = paddle.concat(y_true, axis=0)
            y_true2 = y_true1.numpy()
            y_true_final = y_true2.flatten()
            with open('/home/aistudio/logsCSP_x_NewWave/T-ytrue.pkl', 'wb') as f:
                pickle.dump(y_true_final, f)   

        dev_loss = (total_loss/len(dev_loader))
        dev_score = metric.accumulate() 

        # 记录WaveletAttention相关特征
        if Ispredict and y_down2 is not None:
            with open('/home/aistudio/logsCSP_x_NewWave/T-y_down2.pkl', 'wb') as f:
                pickle.dump(y_down2, f) 
            with open('/home/aistudio/logsCSP_x_NewWave/T-y_down4.pkl', 'wb') as f:
                pickle.dump(y_down4, f) 
            # 若有其他特征则对应保存
            # if y_mid_bn2 is not None:
            #     with open('/home/aistudio/logsCSP_x_NewWave/T-y_mid_bn2.pkl', 'wb') as f:
            #         pickle.dump(y_mid_bn2, f) 
            # # 其余特征类似...
        
        # 记录验证集loss与acc到CSV文件
        csv_file_path = 'logsCSP_x_NewWave/T-dev_epoch_acc_losses.csv'     
        file_is_empty = not os.path.isfile(csv_file_path) or os.stat(csv_file_path).st_size == 0  
        header = ['epoch', 'accuracy', 'loss']  
        with open(csv_file_path, mode='a', newline='') as csv_file:  
            writer = csv.writer(csv_file)  
            if file_is_empty:  
                writer.writerow(header)  
            writer.writerow([epoch, dev_score, dev_loss])

    # 恢复训练模式
    model.train()
    return dev_score, dev_loss

#******************1.数据集加载********************#
from paddle.vision.transforms import transforms
import numpy as np  # 需导入numpy

## 修正后的图像预处理：添加uint8→float32的转换
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    # 核心新增步骤：将PIL图像转为float32数组并归一化到0-1
    lambda img: np.array(img, dtype=np.float32) / 255.0,
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

random.seed(0)
train_txt_file = 'classified_data/train.txt'  # 替换为实际的txt文件路径
val_txt_file = 'classified_data/val.txt'  # 替换为实际的txt文件路径
# 确保使用的是修改后的新数据集类（如果文件名或类名有变化）
train_dataset = CustomDataset(train_txt_file, transform=transform)
val_dataset = CustomDataset(val_txt_file, transform=transform)
print(f'train_custom_dataset images: {len(train_dataset)}, val_custom_dataset images: {len(val_dataset)}')

# 新增代码：创建logsCSP_x_NewWave目录
if not os.path.exists('logsCSP_x_NewWave'):
    os.makedirs('logsCSP_x_NewWave')

#********************2.模型构建***************************#
num_classes = 4          # 实际的类别数

# 修改：导入包含WaveletAttention的模型
from CSP_x_NewWave import CSPDarknetXWithWavelet   # 修改：导入新模型
model = CSPDarknetXWithWavelet(num_classes=num_classes)  # 修改：使用新模型

#********************3.训练*******************************#
paddle.seed(100)

## 超参数设置
learning_rate = 0.0001    # 学习率
batch_size = 8         # 批次大小

# 设置自定义的准确率函数
metric = Accuracy(is_logist=True)
## 设置优化器
optimizer = paddle.optimizer.Adam(learning_rate=learning_rate, parameters=model.parameters())
## 设置损失函数
loss_fn = paddle.nn.CrossEntropyLoss()
## 创建数据加载器
train_data_loader = paddle.io.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_data_loader = paddle.io.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
print("data loaded!")

# 设置迭代次数
start_epoch = 0             # 起始迭代次数
num_epochs = 100              # 迭代次数
global_step = 0             # 总batch数量
num_training_steps = num_epochs * len(train_data_loader)  # 训练总的步数
best_score = 0              # 最佳得分

# 加载当前最新模型，继续训练
model_path = 'logsCSP_x_NewWave/best_model.pdparams'
if os.path.isfile(model_path):
    # 加载模型参数
    model_state_dict = paddle.load(model_path)
    model.set_state_dict(model_state_dict)
    print("成功加载curr_model！")
else:
    print("curr_model不存在，将从头开始训练。")

model.train()
for epoch in range(start_epoch, num_epochs):
    total_loss = 0 # 总损失，为了计算mean loss
    total_acc = 0 # 总准确率

    for batch_id, data in enumerate(train_data_loader):
        
        ## 获取数据集中的图片与label，然后送入model
        x_data, y_data = data
        predicts = model(x_data)    # 预测结果  
        
        ## 计算损失
        loss = loss_fn(predicts, y_data)
        total_loss += loss 

        # 计算准确率
        y_data = paddle.unsqueeze(y_data, axis=1)
        acc = paddle.metric.accuracy(predicts, y_data)
        total_acc += acc
        
        ## 反向传播 
        loss.backward()

        # 打印信息
        if (batch_id+1) % 30 == 0:
            print(f"epoch: {epoch}, batch_id: {batch_id+1}, loss is: {loss.numpy()}, acc is: {acc.numpy()}")

        ## 更新参数 
        optimizer.step()
        ## 梯度清零
        optimizer.clear_grad()

        global_step += 1

    # 计算当前epoch的训练loss和acc
    trn_loss = (total_loss / len(train_data_loader)).item()
    trn_acc = (total_acc / len(train_data_loader)).item()
    
    # 保存训练过程数据到CSV文件
    csv_file_path = 'logsCSP_x_NewWave/T-train_epoch_acc_losses.csv'     
    file_is_empty = not os.path.isfile(csv_file_path) or os.stat(csv_file_path).st_size == 0  
    header = ['epoch', 'accuracy', 'loss']  
    with open(csv_file_path, mode='a', newline='') as csv_file:  
        writer = csv.writer(csv_file)  
        if file_is_empty:  
            writer.writerow(header)  
        writer.writerow([epoch, trn_acc, trn_loss])

    # 保存当前step模型，方便断点续训
    save_path = "logsCSP_x_NewWave/curr_model.pdparams"
    paddle.save(model.state_dict(), save_path)

    # 判断是否需要评价
    if (epoch+1) % 1 == 0 or epoch == 0 or epoch == num_epochs-1 :
        dev_score, dev_loss = evaluate(model=model, metric=metric, dev_loader=val_data_loader, Ispredict=False, epoch=epoch)
        print(f"[Evaluate]  dev score: {dev_score:.5f}, dev loss: {dev_loss:.5f}") 
        # 如果当前指标为最优指标，保存该模型
        if dev_score > best_score:
            paddle.save(model.state_dict(), "logsCSP_x_NewWave/best_model.pdparams")
            print(f"[Evaluate] best accuracy performence has been updated: {best_score:.5f} --> {dev_score:.5f}")
            best_score = dev_score