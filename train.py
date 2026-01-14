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

y_pred = []
y_true = []
y_features_for_tsne = []


def evaluate(model, metric, dev_loader, is_predict, epoch):
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

        logits = model(X)

        loss = loss_fn(logits, y)
        total_loss += loss.item()

        metric.update(logits, y)

        if is_predict:
            y_pred.append(logits.detach())
            y_true.append(y.detach())

            with paddle.no_grad():
                features = model.get_features(X)
                features_flat = model.global_pool(features).flatten(start_axis=1)
                y_features_for_tsne.append(features_flat.detach())

        print(f"IN eval, batch id {batch_id} is completed!")

    if is_predict:
        save_prediction_results()

    dev_loss = total_loss / len(dev_loader) if len(dev_loader) > 0 else 0
    dev_score = metric.accumulate()

    save_dev_results_to_csv(epoch, dev_score, dev_loss)

    return dev_score, dev_loss


def save_prediction_results():
    try:
        all_pred = paddle.concat(y_pred, axis=0)
        all_true = paddle.concat(y_true, axis=0)

        pred_probs = all_pred.numpy()
        pred_classes = np.argmax(pred_probs, axis=-1).flatten()
        true_classes = all_true.numpy().flatten()

        with open('logsUnet_MSDWA/T-ypred.pkl', 'wb') as f:
            pickle.dump(pred_classes, f)
        with open('logsUnet_MSDWA/T-ytrue.pkl', 'wb') as f:
            pickle.dump(true_classes, f)

        if y_features_for_tsne:
            all_features = paddle.concat(y_features_for_tsne, axis=0)
            with open('logsUnet_MSDWA/T-OutputForTsne.pkl', 'wb') as f:
                pickle.dump(all_features.numpy(), f)

        print("预测结果保存成功！")

    except Exception as e:
        print(f"保存预测结果时出错: {e}")


def save_dev_results_to_csv(epoch, accuracy, loss):
    csv_file_path = 'logsUnet_MSDWA/T-dev_epoch_acc_losses.csv'
    file_is_empty = not os.path.isfile(csv_file_path) or os.stat(csv_file_path).st_size == 0

    with open(csv_file_path, mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        if file_is_empty:
            writer.writerow(['epoch', 'accuracy', 'loss'])
        writer.writerow([epoch, accuracy, loss])


def save_train_results_to_csv(epoch, accuracy, loss):
    csv_file_path = 'logsUnet_MSDWA/T-train_epoch_acc_losses.csv'
    file_is_empty = not os.path.isfile(csv_file_path) or os.stat(csv_file_path).st_size == 0

    with open(csv_file_path, mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        if file_is_empty:
            writer.writerow(['epoch', 'accuracy', 'loss'])
        writer.writerow([epoch, accuracy, loss])


def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        lambda x: x.astype('float32') if x.dtype != 'float32' else x,
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


random.seed(0)
paddle.seed(100)

train_txt_file = 'newdata/train.txt'
val_txt_file = 'newdata/val70.txt'

transform = get_transforms()
train_dataset = CustomDataset(train_txt_file, transform=transform)
val_dataset = CustomDataset(val_txt_file, transform=transform)

print(f'train_custom_dataset images: {len(train_dataset)}, test_custom_dataset images: {len(val_dataset)}')

if not os.path.exists('logsUnet_MSDWA'):
    os.makedirs('logsUnet_MSDWA')

num_classes = 4

from Unet_MSDWA import RoadDiseaseClassifier

model = RoadDiseaseClassifier(num_classes=num_classes, use_msdwa=True)

learning_rate = 0.0001
batch_size = 16

metric = Accuracy(is_logist=True)
optimizer = paddle.optimizer.Adam(learning_rate=learning_rate, parameters=model.parameters())
loss_fn = paddle.nn.CrossEntropyLoss()
train_data_loader = paddle.io.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_data_loader = paddle.io.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
print("data loaded!")

start_epoch = 0
num_epochs = 100
global_step = 0
num_training_steps = num_epochs * len(train_data_loader)
best_score = 0

model_path = 'logsUnet_MSDWA/best_model.pdparams'
if os.path.isfile(model_path):
    try:
        model_state_dict = paddle.load(model_path)
        model.set_state_dict(model_state_dict)
        print("成功加载curr_model！")
    except Exception as e:
        print(f"加载模型失败: {e}，将从头开始训练。")
else:
    print("curr_model不存在，将从头开始训练。")

print("开始训练...")
model.train()

for epoch in range(start_epoch, num_epochs):
    total_loss = 0
    total_acc = 0

    for batch_id, data in enumerate(train_data_loader):
        x_data, y_data = data
        predicts = model(x_data)

        loss = loss_fn(predicts, y_data)
        total_loss += loss

        y_data_2d = paddle.unsqueeze(y_data, axis=1)
        acc = paddle.metric.accuracy(predicts, y_data_2d)
        total_acc += acc

        if (batch_id + 1) % 30 == 0:
            print(f"epoch: {epoch}, batch_id: {batch_id + 1}, loss is: {loss.numpy()}, acc is: {acc.numpy()}")

        loss.backward()
        optimizer.step()
        optimizer.clear_grad()

        global_step += 1

    trn_loss = (total_loss / len(train_data_loader)).item() if len(train_data_loader) > 0 else 0
    trn_acc = (total_acc / len(train_data_loader)).item() if len(train_data_loader) > 0 else 0

    print(f"Epoch {epoch} - 训练损失: {trn_loss:.4f}, 训练准确率: {trn_acc:.4f}")

    save_train_results_to_csv(epoch, trn_acc, trn_loss)

    save_path = "logsUnet_MSDWA/curr_model.pdparams"
    paddle.save(model.state_dict(), save_path)

    if (epoch + 1) % 1 == 0 or epoch == 0 or epoch == num_epochs - 1:
        is_predict = (epoch == num_epochs - 1)
        dev_score, dev_loss = evaluate(model=model, metric=metric, dev_loader=val_data_loader, is_predict=is_predict,
                                       epoch=epoch)
        print(f"[Evaluate]