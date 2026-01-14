import os
import random
import cv2
import paddle
import numpy as np
from paddle.io import Dataset
from paddle.vision import transforms
from Mydataset import CustomDataset
from MyAccuracy import Accuracy
import time
import csv  
import pickle

# 验证记录信息的变量定义
y_pred = []
y_true = []
y_features_for_tsne = []

def evaluate(model, metric, dev_loader, is_predict, epoch):
    """评估函数：计算验证集损失和准确率"""
    model.eval()
    total_loss = 0
    metric.reset() 
    
    global y_pred, y_true, y_features_for_tsne
    if is_predict:
        y_pred = []
        y_true = []
        y_features_for_tsne = []
    
    loss_fn = paddle.nn.CrossEntropyLoss()
    
    for batch_id, data in enumerate(dev_loader):
        X, y = data
        
        # 前向传播
        logits = model(X)
        
        # 计算损失
        loss = loss_fn(logits, y)
        total_loss += loss.item()
        
        # 更新评价指标
        metric.update(logits, y)
        
        # 记录预测结果（用于混淆矩阵等分析）
        if is_predict:
            y_pred.append(logits.detach())
            y_true.append(y.detach())
            
            # 提取特征用于t-SNE可视化
            with paddle.no_grad():
                features = model.get_features(X)  # 使用模型的get_features方法
                # 转换为一维特征向量
                features_flat = model.global_pool(features).flatten(start_axis=1)
                y_features_for_tsne.append(features_flat.detach())
        
        print(f"IN eval, batch id {batch_id} is completed!")
        
    # 保存预测结果和特征
    if is_predict:
        save_prediction_results()
    
    # 计算平均损失和准确率
    dev_loss = total_loss / len(dev_loader) if len(dev_loader) > 0 else 0
    dev_score = metric.accumulate()
    
    # 记录到CSV
    save_dev_results_to_csv(epoch, dev_score, dev_loss)
    
    return dev_score, dev_loss

def save_prediction_results():
    """保存预测结果用于后续分析"""
    try:
        # 合并所有batch的预测结果
        all_pred = paddle.concat(y_pred, axis=0)
        all_true = paddle.concat(y_true, axis=0)
        
        # 转换为numpy并保存
        pred_probs = all_pred.numpy()  # 保存概率
        pred_classes = np.argmax(pred_probs, axis=-1).flatten()  # 预测类别
        true_classes = all_true.numpy().flatten()  # 真实类别
        
        # 保存预测结果
        with open('logsUnet_MSDWA/T-ypred.pkl', 'wb') as f:
            pickle.dump(pred_classes, f)
        with open('logsUnet_MSDWA/T-ytrue.pkl', 'wb') as f:
            pickle.dump(true_classes, f)
        
        # 保存特征用于t-SNE
        if y_features_for_tsne:
            all_features = paddle.concat(y_features_for_tsne, axis=0)
            with open('logsUnet_MSDWA/T-OutputForTsne.pkl', 'wb') as f:
                pickle.dump(all_features.numpy(), f)
                
        print("预测结果保存成功！")
        
    except Exception as e:
        print(f"保存预测结果时出错: {e}")

def save_dev_results_to_csv(epoch, accuracy, loss):
    """保存验证结果到CSV"""
    csv_file_path = 'logsUnet_MSDWA/T-dev_epoch_acc_losses.csv'
    file_is_empty = not os.path.isfile(csv_file_path) or os.stat(csv_file_path).st_size == 0
    
    with open(csv_file_path, mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        if file_is_empty:
            writer.writerow(['epoch', 'accuracy', 'loss'])
        writer.writerow([epoch, accuracy, loss])

def save_train_results_to_csv(epoch, accuracy, loss):
    """保存训练结果到CSV"""
    csv_file_path = 'logsUnet_MSDWA/T-train_epoch_acc_losses.csv'
    file_is_empty = not os.path.isfile(csv_file_path) or os.stat(csv_file_path).st_size == 0
    
    with open(csv_file_path, mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        if file_is_empty:
            writer.writerow(['epoch', 'accuracy', 'loss'])
        writer.writerow([epoch, accuracy, loss])

#******************1.数据集加载********************#
def get_transforms():
    """获取数据预处理变换"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # 确保数据类型为float32
        lambda x: x.astype('float32') if x.dtype != 'float32' else x,
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

# 设置随机种子
random.seed(0)
paddle.seed(100)

# 数据路径
train_txt_file = 'newdata/train.txt'  # 替换为实际的txt文件路径
val_txt_file = 'newdata/val70.txt'      # 替换为实际的txt文件路径

# 创建数据集
transform = get_transforms()
train_dataset = CustomDataset(train_txt_file, transform=transform)
val_dataset = CustomDataset(val_txt_file, transform=transform)

print(f'train_custom_dataset images: {len(train_dataset)}, test_custom_dataset images: {len(val_dataset)}')

# 创建logsUnet_MSDWA目录
if not os.path.exists('logsUnet_MSDWA'):
    os.makedirs('logsUnet_MSDWA')

#********************2.模型构建***************************#
num_classes = 4  # 实际的类别数

from Unet_MSDWA import RoadDiseaseClassifier
model = RoadDiseaseClassifier(num_classes=num_classes, use_msdwa=True)

#********************3.训练*******************************#
# 超参数设置
learning_rate = 0.0001    # 学习率
batch_size = 16           # 批次大小

# 设置自定义的准确率函数
metric = Accuracy(is_logist=True)
# 设置优化器
optimizer = paddle.optimizer.Adam(learning_rate=learning_rate, parameters=model.parameters())
# 设置损失函数
loss_fn = paddle.nn.CrossEntropyLoss()
# 创建数据加载器
train_data_loader = paddle.io.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_data_loader = paddle.io.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
print("data loaded!")

# 设置迭代次数
start_epoch = 0             # 起始迭代次数
num_epochs = 100            # 迭代次数
global_step = 0             # 总batch数量
num_training_steps = num_epochs * len(train_data_loader)  # 训练总的步数
best_score = 0              # 最佳得分

# 加载当前最新模型，继续训练
model_path = 'logsUnet_MSDWA/best_model.pdparams'
if os.path.isfile(model_path):
    try:
        # 加载模型参数
        model_state_dict = paddle.load(model_path)
        model.set_state_dict(model_state_dict)
        print("成功加载curr_model！")
    except Exception as e:
        print(f"加载模型失败: {e}，将从头开始训练。")
else:
    print("curr_model不存在，将从头开始训练。")

# 开始训练
print("开始训练...")
model.train()

for epoch in range(start_epoch, num_epochs):
    total_loss = 0  # 总损失，为了计算mean loss
    total_acc = 0   # 总准确率

    for batch_id, data in enumerate(train_data_loader):
        # 获取数据集中的图片与label，然后送入model
        x_data, y_data = data
        predicts = model(x_data)    # 预测结果  
        
        # 计算损失
        loss = loss_fn(predicts, y_data)
        total_loss += loss 
        
        # 计算准确率（需要将1维标签转为2维）
        y_data_2d = paddle.unsqueeze(y_data, axis=1)
        acc = paddle.metric.accuracy(predicts, y_data_2d)
        total_acc += acc
        
        # 反向传播 
        loss.backward()

        # 打印信息
        if (batch_id+1) % 30 == 0:
            print(f"epoch: {epoch}, batch_id: {batch_id+1}, loss is: {loss.numpy()}, acc is: {acc.numpy()}")

        # 更新参数 
        optimizer.step()
        # 梯度清零
        optimizer.clear_grad()

        global_step += 1

    # 计算epoch粒度的损失和准确率
    trn_loss = (total_loss / len(train_data_loader)).item() if len(train_data_loader) > 0 else 0
    trn_acc = (total_acc / len(train_data_loader)).item() if len(train_data_loader) > 0 else 0

    print(f"Epoch {epoch} - 训练损失: {trn_loss:.4f}, 训练准确率: {trn_acc:.4f}")

    # 记录训练数据到CSV
    save_train_results_to_csv(epoch, trn_acc, trn_loss)

    # 保存当前模型，方便断点续训
    save_path = "logsUnet_MSDWA/curr_model.pdparams"
    paddle.save(model.state_dict(), save_path)

    # 评价模型
    if (epoch+1) % 1 == 0 or epoch == 0 or epoch == num_epochs-1:
        # 最后一个epoch保存预测结果用于分析
        is_predict = (epoch == num_epochs - 1)
        dev_score, dev_loss = evaluate(model=model, metric=metric, dev_loader=val_data_loader, is_predict=is_predict, epoch=epoch)
        print(f"[Evaluate]  dev score: {dev_score:.5f}, dev loss: {dev_loss:.5f}") 
        model.train()
        
        # 保存最佳模型
        if dev_score > best_score:
            paddle.save(model.state_dict(), "logsUnet_MSDWA/best_model.pdparams")
            print(f"[Evaluate] best accuracy updated: {best_score:.5f} --> {dev_score:.5f}")
            best_score = dev_score

print("训练完成！")
print(f"最佳验证准确率: {best_score:.4f}")