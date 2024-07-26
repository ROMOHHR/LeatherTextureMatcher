import re
import os
import configparser
import json

import torch
import torch.nn as nn
import numpy as np
from torchvision import models
from torch.utils.data.sampler import WeightedRandomSampler

# 自定义模块
from optimizer import MyOptimizer
from metrics import AccuracyScore
from dataset_utils import DataSet_Default


torch.set_printoptions(precision=2, sci_mode=False)


class TextureClassifier_Default:
    def __init__(self, model, lr=0.05, weight_decay=0.1):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss()
        self.acc_fn = AccuracyScore()
        self.opt = MyOptimizer(
            parameters=[p for p in self.model.parameters() if p.requires_grad is True],
            learning_rate=lr,
            weight_decay=weight_decay
        ).curr_optim()
        
        # 加载模型最佳参数
        self.current_epoch = 0
        self.model_dir = 'models'
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        else:
            names = os.listdir(self.model_dir)
            names = list(filter(lambda x: re.match(r'(model_\d{4}|best).pth', x) != None, names))
            if len(names) > 0:
                names.sort()
                name = names[-1]
                if name != 'best.pth':
                    self.current_epoch = int(name[-8:-4])
                missing_keys, unexpected_keys = self.model.load_state_dict(
                    torch.load(os.path.join(self.model_dir, name)))
        self.model = self.model.to(self.device)  # 注意这一行要放在后面

        # 加载模型最佳参数的测试正确率 self.best_test_acc
        self.best_test_acc = 0
        self.config_dir = os.path.join(self.model_dir, 'config.ini')
        if not os.path.exists(self.config_dir):
            config = configparser.ConfigParser()
            config.read(self.config_dir)
            config['model'] = {'best_test_acc': self.best_test_acc,
                               'train_acc': 0,
                               'test_acc': 0}
            config.write(open(self.config_dir, 'w'))
        else:
            config = configparser.ConfigParser()
            config.read(self.config_dir)
            self.best_test_acc = config.getfloat('model', 'best_test_acc')# 读取数据库配置

    def save_model(self, epoch):  # 模型保存
        model_path = os.path.join(self.model_dir, f"model_{epoch:04d}.pth")
        torch.save(self.model.state_dict(), model_path)

    def save_best_model(self, train_acc, test_acc):
        acc = 0.1 * train_acc + 0.9 * test_acc
        if self.best_test_acc <= acc:  # 等于的时候也更新
            self.best_test_acc = acc
            model_path = os.path.join(self.model_dir, "best.pth")
            torch.save(self.model.state_dict(), model_path)

            config = configparser.ConfigParser()
            config.read(self.config_dir)
            config.set('model', 'best_test_acc', str(self.best_test_acc))  # 修改数据库配置
            config.set('model', 'train_acc', str(train_acc))
            config.set('model', 'test_acc', str(test_acc))
            config.write(open(self.config_dir, 'w'))

    def train(self, train_data_dir, test_data_dir, batch_size=50, total_epoch=50, num_workers=0, print_interval=1, mode=0):
        # 处理样本不平衡问题（0:28  1:125  2:152） total=305 => 0:(1/(28/305))=10.89  1:(1/(125/305))=2.44  2:(1/(152/305))=2
        # weights = [10.89 if label == 0  else 2.44 if label == 1 else 2 for data, label in self.dataset]
        # sampler = WeightedRandomSampler(weights, num_samples=len(self.dataset), replacement=True)

        # 1. 加载数据
        trainset = DataSet_Default(root_dir=train_data_dir,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              istrainning=True,
                              sampler=None,
                              shuffle=True)
        testset = DataSet_Default(root_dir=test_data_dir,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             istrainning=False,
                             sampler=None,
                             shuffle=False)
        
        if mode == 1:  # 保存类别及类别索引
            dict_class_idx = trainset.dataset.class_to_idx
            dict_class_idx = dict(zip(dict_class_idx.values(), dict_class_idx.keys()))

            config = configparser.ConfigParser()
            config.read(self.config_dir)
            config['class_idx'] = {"Total":str(len(dict_class_idx)), "class_detail":json.dumps(dict_class_idx)}
            config.write(open(self.config_dir, 'w'))

        
        for epoch in range(self.current_epoch, total_epoch):
            self.model.train(True)  # Sets the module in training mode.

            output_, labels_ = torch.tensor([]).to(self.device), torch.tensor([]).to(self.device)
            batch = 0
            for inputs, labels in trainset:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # forward
                output = self.model(inputs)
                # output = output.logits
                loss = self.loss_fn(output, labels)

                # backward
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                acc = self.acc_fn(output, labels)

                output_ = torch.cat([output_, output])
                labels_ = torch.cat([labels_, labels])

                if batch % print_interval == 0:
                    print(f'{epoch + 1}/{total_epoch} {batch} train_loss={loss.item():.4f} -- acc={acc.item():.4f}')
                    batch += 1
            train_loss = self.loss_fn(output_, labels_.to(dtype=torch.long))
            train_acc = self.acc_fn(output_, labels_.to(dtype=torch.long))
            print(f'train total_loss {train_loss:.4f}, train total_acc {train_acc:.4f}')


            output_, labels_ = torch.tensor([]).to(self.device), torch.tensor([]).to(self.device)
            batch = 0
            with torch.no_grad():
                for inputs, labels in testset:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # forward
                    output = self.model(inputs)
                    # output = output.logits
                    loss = self.loss_fn(output, labels)
                    acc = self.acc_fn(output, labels)

                    output_ = torch.cat([output_, output])
                    labels_ = torch.cat([labels_, labels])

                    if batch % print_interval == 0:
                        print(f'{epoch + 1}/{total_epoch} {batch} test_loss={loss.item():.4f} --acc={acc.item():.4f}')
                    batch += 1
            test_loss = self.loss_fn(output_, labels_.to(dtype=torch.long))
            test_acc = self.acc_fn(output_, labels_.to(dtype=torch.long))
            print(f'test total_loss {test_loss:.4f}, test total_acc {test_acc:.4f}\n')

            self.save_model(epoch + 1)
            self.save_best_model(train_acc.numpy(), test_acc.numpy())


from torchvision.models.inception import Inception3, BasicConv2d
from torchvision.models import ResNet18_Weights, ResNet50_Weights, VGG11_Weights
if __name__ == '__main__':
    m = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    m.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    m.fc = nn.Linear(in_features=512, out_features=2, bias=True)

    # m = models.vgg11_bn(weights=VGG11_BN_Weights.IMAGENET1K_V1)
    # m.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    # m.classifier[6] = nn.Linear(in_features=4096, out_features=2, bias=True)

    # m = models.vgg11(weights=VGG11_Weights.IMAGENET1K_V1)
    # m.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    # m.classifier[6] = nn.Linear(in_features=4096, out_features=2, bias=True)

    # m = models.vgg13_bn(weights=VGG13_BN_Weights.DEFAULT)
    # m.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    # m.classifier[6] = nn.Linear(in_features=4096, out_features=5, bias=True)

    # m = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    # m.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    # m.fc = nn.Linear(in_features=2048, out_features=40, bias=True)

    if 0:
        print(m)
    elif 0:
        # 保存模型
        m.load_state_dict(torch.load(os.path.join('models', 'best.pth')))
        torch.save(m, os.path.join('models', 'best_model.pth'))
    else:
        batch_size = 120
        lr = 0.005
        total_epoch = 50
        weight_decay = 0 #0.2

        # train_data_dir = './datasets/data3/output/train'
        # test_data_dir = './datasets/data3/output/test'
        train_data_dir = './datasets/data5/train'
        test_data_dir = './datasets/data5/test'
        model = TextureClassifier_Default(m, lr, weight_decay)
        model.train(train_data_dir, test_data_dir, batch_size, total_epoch, mode=1)


