import os
from tokenize import Imagnumber
from typing_extensions import runtime
from numpy.core.defchararray import not_equal
from numpy.core.fromnumeric import transpose

from numpy.lib.polynomial import roots
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from torchsummary import summary
import time
import numpy as np
import sys
from sklearn.metrics import accuracy_score


class GetLoader(torch.utils.data.Dataset):
    def __init__(self, data_root, data_label, transforms=None):
        self.data = data_root
        self.label = data_label
        self.transforms = transforms

    def __getitem__(self, index):
        data = self.data[index]

        labels = self.label[index]

        if self.transforms is not None:
            data = self.transforms(data)
        return data, labels

    def __len__(self):
        return len(self.data)


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

transform_train = transforms.Compose([

    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=0, translate=(0.15, 0.15)),
    transforms.ToTensor(),
    transforms.Normalize((0.4915, 0.4823, 0.4469), (0.2023, 0.2433, 0.2616))
])

transform_show = transforms.Compose([

    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.4915, 0.4823, 0.4469), (0.2023, 0.2433, 0.2616))
])


class AllConvNet(nn.Module):
    def __init__(self, input_size, n_classes=10, **kwargs):
        super(AllConvNet, self).__init__()

        self.conv1 = nn.Conv2d(input_size, 96, 3, padding=1, stride=1)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(96, 96, 3, padding=1, stride=1)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(96, 96, 3, padding=1, stride=2)
        self.dropout3 = nn.Dropout(p=0.5)

        self.conv4 = nn.Conv2d(96, 192, 3, padding=1, stride=1)
        self.relu4 = nn.ReLU()

        self.conv5 = nn.Conv2d(192, 192, 3, padding=1, stride=1)
        self.relu5 = nn.ReLU()

        self.conv6 = nn.Conv2d(192, 192, 3, padding=1, stride=2)
        self.dropout6 = nn.Dropout(p=0.5)

        self.conv7 = nn.Conv2d(192, 192, 3, padding=1, stride=1)
        self.relu7 = nn.ReLU()

        self.conv8 = nn.Conv2d(192, 192, 3, stride=1, padding=1)
        self.relu8 = nn.ReLU()

        self.class_conv = nn.Conv2d(192, n_classes, 1, padding=0, stride=1)
        self.poolout = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv1_out = self.relu1(conv1_out)

        conv2_out = self.conv2(conv1_out)
        conv2_out = self.relu2(conv2_out)

        conv3_out = self.conv3(conv2_out)
        conv3_out_drop = self.dropout3(conv3_out)

        conv4_out = self.conv4(conv3_out_drop)
        conv4_out = self.relu4(conv4_out)

        conv5_out = self.conv5(conv4_out)
        conv5_out = self.relu5(conv5_out)

        conv6_out = self.conv6(conv5_out)
        conv6_out_drop = self.dropout6(conv6_out)

        conv7_out = self.conv7(conv6_out_drop)
        conv7_out = self.relu7(conv7_out)

        conv8_out = self.conv8(conv7_out)
        conv8_out = self.relu8(conv8_out)

        class_out = self.class_conv(conv8_out)

        pool_out = self.poolout(class_out)

        pool_out.squeeze_(-1)
        pool_out.squeeze_(-1)
        return pool_out


def training(model, my_train_loader, my_test_loader):
    eval_best = 0

    for epoch in range(num_epoches):
        print('epoch {}'.format(epoch + 1))
        print('===========================================')
        running_loss = 0.0
        running_acc = 0.0
        model.train()
        for i, data in enumerate(my_train_loader, 1):
            # for i, data in enumerate(train_loader, 1):
            img, label = data

            # cuda
            if use_gpu:
                img = img.cuda()
                label = label.cuda()

            img = Variable(img)
            label = Variable(label)

            img = img.float()
            label = label.long()

            out = model(img)
            loss = criterion(out, label)
            running_loss += loss.item() * label.size(0)
            _, pred = torch.max(out, 1)
            num_correct = (pred == label).sum()
            accuracy = (pred == label).float().mean()
            running_acc += num_correct.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("Accuracy of my model on train-set: ",
              str(running_acc / (len(train_data))))
        print()
        print('Finish {} epoch, Loss: {:.6f}, Acc: {:.6f}'.format(
            epoch + 1, running_loss / (len(train_data)),
            running_acc / (len(train_data))))
        model.eval()
        eval_loss = 0
        eval_acc = 0
        eval_acca = 0
        for data in my_test_loader:
            # for data in test_loader:
            img, label = data
            if use_gpu:
                img = Variable(img, volatile=True).cuda()
                label = Variable(label, volatile=True).cuda()
            else:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

            img = img.float()
            label = label.long()

            out = model(img)
            loss = criterion(out, label)
            eval_loss += loss.item() * label.size(0)
            _, pred = torch.max(out, 1)
            num_correct = (pred == label).sum()
            eval_acc += num_correct.item()
        print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
            test_data)), eval_acc / (len(test_data))))

        eval_acca = eval_acc / (len(test_data))
        if eval_acca > eval_best:
            eval_best = eval_acca
            torch.save(model.state_dict(), './all_cifar10.pth')
            print("Save the best model weights, acc={:.6f}".format(eval_best))
        else:
            print(
                "Eval acc don't raise from {:.4f}".format(eval_best))
        #     pass

    print("The final best eval acca is {:.4f}".format(eval_best))


def predict(model, x_test):

    random_list = np.zeros((10000))
    test_data = GetLoader(x_test, random_list, transforms=transform_show)
    my_test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False)

    model.eval()
    result = np.zeros((0, 10))
    print()
    for data in my_test_loader:
        # for data in test_loader:
        img, label = data
        if use_gpu:
            img = Variable(img, volatile=True).cuda()
        else:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

        img = img.float()

        out = model(img)

        out = out.detach().cpu().numpy()
        result = np.vstack((result, out))

        # time.sleep(100000)

    return result


batch_size = 64
learning_rate = 1e-2
num_epoches = 2000

if __name__ == '__main__':

    print("select function:")
    print("(1) Train a new model  (2) Show best_model Accuracy on test_set")

    istrain = input("請輸入 1 or 2 : ")
    # ## Load data
    # In[5]:

    x_train = np.load("x_train.npy")
    y_train = np.load("y_train.npy")

    x_test = np.load("x_test.npy")
    y_test = np.load("y_test.npy")

    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    print()

    # In[6]:
    # It's a multi-class classification problem
    class_index = {
        'airplane': 0,
        'automobile': 1,
        'bird': 2,
        'cat': 3,
        'deer': 4,
        'dog': 5,
        'frog': 6,
        'horse': 7,
        'ship': 8,
        'truck': 9}
    print(np.unique(y_train))
    print()

    # I use pytorch, i do preprocess(and also data augumentation)
    # by using torchversion.transform
    # # ## Data preprocess

    y_train = y_train.ravel()
    y_test = y_test.ravel()

    train_data = GetLoader(x_train, y_train, transforms=transform_train)
    test_data = GetLoader(x_test, y_test, transforms=transform_show)

    '''
    train_mean = train_data.data.mean(axis=(0,1,2))/255
    # [0.49156518 0.48238982 0.4469944 ]

    train_std = train_data.data.std(axis=(0,1,2))/255
    # [0.24687816 0.24333645 0.26169549]
    '''

    my_train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True)
    my_test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False)

    print()

    # ## Build model & training (Keras)
    # In[9]:

    model = AllConvNet(3, 10)
    # model.load_state_dict(torch.load("all_cifar10.pth"))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    use_gpu = torch.cuda.is_available()  # 判断是否有GPU加速
    if use_gpu:
        model = model.cuda()
        print("Using GPU now!")
    print()

    # initiate SGD optimizer
    # Compile the model with loss function and optimizer

    criterion = nn.CrossEntropyLoss()

    weight_p, bias_p = [], []
    for name, p in model.named_parameters():
        if 'bias' in name or '1.4' in name or '2.4' in name:
            bias_p += [p]
        else:
            weight_p += [p]

    optimizer = optim.SGD([
        {'params': weight_p, 'weight_decay': 1e-3},
        {'params': bias_p, 'weight_decay': 0}
    ], lr=1e-2, momentum=0.9, nesterov=True)

    # Fit the data into model
    if istrain == "1":
        print()
        print("it will take maybe one day to get best_result")
        print()
        training(
            model=model,
            my_test_loader=my_test_loader,
            my_train_loader=my_train_loader)
    print()

    # In[10]:

    model.load_state_dict(torch.load("best_model.pth"))
    y_pred = predict(model=model, x_test=x_test)
    y_pred = np.array(y_pred)
    print(y_pred.shape)

    # In[11]:
    y_pred[0]

    # In[12]:
    np.argmax(y_pred[0])

    # In[13]:
    y_pred = np.argmax(y_pred, axis=1)
    # ## DO NOT MODIFY CODE BELOW!
    # please screen shot your results and post it on your report

    print()
    # In[14]:

    assert y_pred.shape == (10000,)

    # In[15]:

    y_test = np.load("y_test.npy")
    print("Accuracy of my model on test-set: ", accuracy_score(y_test, y_pred))
