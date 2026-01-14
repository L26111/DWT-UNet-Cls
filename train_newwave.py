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

y_pred = None
y_true = None
y_outputForTsne = None
y_down2 = None
y_down4 = None
y_mid_bn2 = None
y_up3 = None
y_up1 = None


def evaluate(model, metric, dev_loader, Ispredict, epoch):
    global y_pred, y_true, y_outputForTsne, y_down2, y_down4, y_mid_bn2, y_up3, y_up1
    y_pred = []
    y_true = []
    y_outputForTsne = None
    y_down2 = None
    y_down4 = None
    y_mid_bn2 = None
    y_up3 = None
    y_up1 = None

    model.eval()
    with paddle.no_grad():
        total_loss = 0
        metric.reset()

        for batch_id, data in enumerate(dev_loader):
            X, y = data
            logits = model(X)

            if Ispredict:
                y_pred.append(logits)
                y_true.append(y)

            loss_fn = paddle.nn.CrossEntropyLoss()
            loss = loss_fn(logits, y).item()
            total_loss += loss

            metric.update(logits, y)

            current_fc1 = model.y_fc1
            if batch_id == 0:
                y_outputForTsne = current_fc1.numpy()
            else:
                y_outputForTsne = np.vstack((y_outputForTsne, current_fc1.numpy()))

            if batch_id == 0:
                y_down2 = model.down2_out.numpy()
                y_down4 = model.down4_out.numpy()
            else:
                y_down2 = np.vstack((y_down2, model.down2_out.numpy()))
                y_down4 = np.vstack((y_down4, model.down4_out.numpy()))

            print(f"IN eval, batch id {batch_id} is completed!")

        import pickle
        if Ispredict:
            with open('/home/aistudio/logsCSP_x_NewWave/T-OutputForTsne.pkl', 'wb') as f:
                pickle.dump(y_outputForTsne, f)

            y_pred1 = paddle.concat(y_pred, axis=0)
            y_pred2 = y_pred1.numpy()
            y_pred3 = np.argmax(y_pred2, axis=-1)
            y_pred_final = y_pred3.flatten()
            with open('/home/aistudio/logsCSP_x_NewWave/T-ypred.pkl', 'wb') as f:
                pickle.dump(y_pred_final, f)

            y_true1 = paddle.concat(y_true, axis=0)
            y_true2 = y_true1.numpy()
            y_true_final = y_true2.flatten()
            with open('/home/aistudio/logsCSP_x_NewWave/T-ytrue.pkl', 'wb') as f:
                pickle.dump(y_true_final, f)

        dev_loss = (total_loss / len(dev_loader))
        dev_score = metric.accumulate()

        if Ispredict and y_down2 is not None:
            with open('/home/aistudio/logsCSP_x_NewWave/T-y_down2.pkl', 'wb') as f:
                pickle.dump(y_down2, f)
            with open('/home/aistudio/logsCSP_x_NewWave/T-y_down4.pkl', 'wb') as f:
                pickle.dump(y_down4, f)

        csv_file_path = 'logsCSP_x_NewWave/T-dev_epoch_acc_losses.csv'
        file_is_empty = not os.path.isfile(csv_file_path) or os.stat(csv_file_path).st_size == 0
        header = ['epoch', 'accuracy', 'loss']
        with open(csv_file_path, mode='a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            if file_is_empty:
                writer.writerow(header)
            writer.writerow([epoch, dev_score, dev_loss])

    model.train()
    return dev_score, dev_loss


from paddle.vision.transforms import transforms
import numpy as np

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    lambda img: np.array(img, dtype=np.float32) / 255.0,
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

random.seed(0)
train_txt_file = 'classified_data/train.txt'
val_txt_file = 'classified_data/val.txt'
train_dataset = CustomDataset(train_txt_file, transform=transform)
val_dataset = CustomDataset(val_txt_file, transform=transform)
print(f'train_custom_dataset images: {len(train_dataset)}, val_custom_dataset images: {len(val_dataset)}')

if not os.path.exists('logsCSP_x_NewWave'):
    os.makedirs('logsCSP_x_NewWave')

num_classes = 4

from CSP_x_NewWave import CSPDarknetXWithWavelet

model = CSPDarknetXWithWavelet(num_classes=num_classes)

paddle.seed(100)

learning_rate = 0.0001
batch_size = 8

metric = Accuracy(is_logist=True)
optimizer = paddle.optimizer.Adam(learning_rate=learning_rate, parameters=model.parameters())
loss_fn = paddle.nn.CrossEntropyLoss()
train_data_loader = paddle.io.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_data_loader = paddle.io.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
print("data loaded!")

start_epoch = 0
num_epochs = 100
global_step = 0
num_training_steps = num_epochs * len(train_data_loader)
best_score = 0

model_path = 'logsCSP_x_NewWave/best_model.pdparams'
if os.path.isfile(model_path):
    model_state_dict = paddle.load(model_path)
    model.set_state_dict(model_state_dict)
    print("成功加载curr_model！")
else:
    print("curr_model不存在，将从头开始训练。")

model.train()
for epoch in range(start_epoch, num_epochs):
    total_loss = 0
    total_acc = 0

    for batch_id, data in enumerate(train_data_loader):
        x_data, y_data = data
        predicts = model(x_data)

        loss = loss_fn(predicts, y_data)
        total_loss += loss

        y_data = paddle.unsqueeze(y_data, axis=1)
        acc = paddle.metric.accuracy(predicts, y_data)
        total_acc += acc

        loss.backward()

        if (batch_id + 1) % 30 == 0:
            print(f"epoch: {epoch}, batch_id: {batch_id + 1}, loss is: {loss.numpy()}, acc is: {acc.numpy()}")

        optimizer.step()
        optimizer.clear_grad()

        global_step += 1

    trn_loss = (total_loss / len(train_data_loader)).item()
    trn_acc = (total_acc / len(train_data_loader)).item()

    csv_file_path = 'logsCSP_x_NewWave/T-train_epoch_acc_losses.csv'
    file_is_empty = not os.path.isfile(csv_file_path) or os.stat(csv_file_path).st_size == 0
    header = ['epoch', 'accuracy', 'loss']
    with open(csv_file_path, mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        if file_is_empty:
            writer.writerow(header)
        writer.writerow([epoch, trn_acc, trn_loss])

    save_path = "logsCSP_x_NewWave/curr_model.pdparams"
    paddle.save(model.state_dict(), save_path)

    if (epoch + 1) % 1 == 0 or epoch == 0 or epoch == num_epochs - 1:
        dev_score, dev_loss = evaluate(model=model, metric=metric, dev_loader=val_data_loader, Ispredict=False,
                                       epoch=epoch)
        print(f"[Evaluate]  dev score: {dev_score:.5f}, dev loss: {dev_loss:.5f}")
        if dev_score > best_score:
            paddle.save(model.state_dict(), "logsCSP_x_NewWave/best_model.pdparams")
            print(f"[Evaluate] best accuracy performence has been updated: {best_score:.5f} --> {dev_score:.5f}")
            best_score = dev_score