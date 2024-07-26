
import os
import numpy as np
import configparser
import json

from PIL import Image
import torch
import torch.nn as nn
from metrics import AccuracyScore
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader

from dataset_utils import DataSet_Default
# from modules.googlenet_v4 import GoogLeNetV4


np.set_printoptions(precision=4, suppress=True)


def softmax(z):
    # 计算softmax函数
    e_z = np.exp(z - np.max(z))
    return e_z / np.sum(e_z)

def convert_img(img):
    return img.convert("L") # 转灰度图 img.convert("RGB")

def get_transforms():
    return transforms.Compose([
                transforms.Lambda(convert_img),
                transforms.ToTensor(),
                transforms.Resize(size=(224, 224), antialias=True)
            ])

def single_predict(model, dict_class_idx, imgpath, topk=2):
    ts = get_transforms()
    img = Image.open(imgpath).convert("L")
    img = ts(img)
    img = img[None, ...]
    r = model(img)
    
    # 获取tensor中前k个最大值及其索引
    values, indices = torch.topk(r[0], k=topk, dim=0, largest=True)
    print("Top k values:")
    print(values)
    print(softmax(values.detach().numpy()))
    print("Their indices:")
    print(indices)
    print([dict_class_idx.get(str(key), None) for key in indices.numpy()])
    
def multi_predict(model, dict_class_idx, topk=2):
    # 加载数据集管理器
    batch_size = 120
    num_workers = 4
    test_data_dir = './datasets/data5/test'

    ts = get_transforms()
    
    dataset = datasets.ImageFolder(root=test_data_dir, transform=ts)
    testset = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
            prefetch_factor=None if num_workers == 0 else num_workers * batch_size)
    
    loss_fn = nn.CrossEntropyLoss()
    acc_fn = AccuracyScore()
    
    # 开始预测
    output_, labels_ = torch.tensor([]).to(device), torch.tensor([]).to(device)
    with torch.no_grad():
        for inputs, labels in testset:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            output = model(inputs)
            output_ = torch.cat([output_, output])
            labels_ = torch.cat([labels_, labels])
            
            for i in range(len(output)):
                print("===== output: =====")
                values, indices = torch.topk(output[i], k=topk, dim=0, largest=True)
                print("Top k values:", softmax(values.detach().numpy()))
                print("Their indices:", indices)
                print([dict_class_idx.get(str(key), None) for key in indices.numpy()])

                print("===== labels: =====")
                print(dict_class_idx.get(str(labels[i].numpy()), None))
                print("")
            
            loss = loss_fn(output, labels)
            acc = acc_fn(output, labels)
            print(f'test loss {loss:.4f}, test acc {acc:.4f}\n')
        
        total_loss = loss_fn(output_, labels_.to(dtype=torch.long))
        total_acc = acc_fn(output_, labels_.to(dtype=torch.long))
        # print("output:\n", output_)
        # print("labels:\n", labels_)
        print(f'test total_loss {total_loss:.4f}, test total_acc {total_acc:.4f}\n')
        
        

def load_model(model_dir, device, mode=0):
    if mode == 0:  # 加载训练好的模型(带参数)
        modelpath = os.path.join(model_dir, 'best_model.pth')
        model = torch.load(modelpath)
        model.to(device)
        model.eval()
        return model
    else:  # 构建模型，加载参数
        # m = models.vgg11_bn()
        # m.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # m.classifier[6] = nn.Linear(in_features=4096, out_features=5, bias=True)

        # m = models.googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1)
        # m.fc = nn.Linear(in_features=1024, out_features=5, bias=True)

        model = models.resnet18()
        model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc = nn.Linear(in_features=512, out_features=40, bias=True)

        # model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        # model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # model.fc = nn.Linear(in_features=2048, out_features=40, bias=True)
        
        # 加载模型权重参数
        modelpath = os.path.join(model_dir, 'best.pth')
        model.load_state_dict(torch.load(modelpath))
        model.to(device)
        model.eval()
        return model


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model_dir = os.path.join('models', 'resnet18_c40_0.9889')
    model_dir = os.path.join('models', 'resnet18_c2_0.8525')
    # model_dir = os.path.join('models')

    # 获取分类信息（分类数量、索引对应的种类）
    configpath = os.path.join(model_dir, 'config.ini')
    config = configparser.ConfigParser()
    config.read(configpath)
    class_detail = config.get('class_idx', 'class_detail')# 读取数据库配置
    dict_class_idx = json.loads(class_detail)

    model = load_model(model_dir, device, mode=0)

    if 1:
        imgpath = r'C:\Users\16428\Desktop\IMG_20240118_171442.jpg'
        # imgpath = os.path.join('datasets', 'data1', 'valid', 'Leather_texture_B_2', '0002_jpg.rf.9d329140d8b7221a88deb20424791ad6.jpg')
        # imgpath = os.path.join('datasets', 'data3', 'output', 'Leather_texture_S_9', '0003.jpg')

        single_predict(model, dict_class_idx, imgpath, topk=2)
    else:
        multi_predict(model, dict_class_idx, topk=2)

