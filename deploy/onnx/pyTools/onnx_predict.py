
import torch
import os
import numpy as np
import onnxruntime
import configparser
import json
from PIL import Image
import numpy as np
from torchvision import transforms

np.set_printoptions(precision=4, suppress=True)


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def batch_softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

def convert_img(img):
    return img.convert("L") # 转灰度图 img.convert("RGB")

def get_transforms():
    return transforms.Compose([
                transforms.Lambda(convert_img),
                transforms.ToTensor(),
                transforms.Resize(size=(224, 224), antialias=True)
            ])

if __name__ == '__main__':
    model_dir = os.path.join('models', 'resnet18_c40_0.9889')

    model_path = os.path.join(model_dir, "TextureClassifierModel.onnx")
    sess = onnxruntime.InferenceSession(model_path)

    imgpath = r'D:\Files\MyProjects\ICTL_TextureMatch\datasets\data3\output\Leather_texture_S_7\0000.jpg'
    ts = get_transforms()
    input_data = Image.open(imgpath).convert("L")
    input_data = ts(input_data)
    input_data = input_data[None, ...]

    output = sess.run(None, {'input': input_data.numpy()})

    # 获取分类信息（分类数量、索引对应的种类）
    configpath = os.path.join(model_dir, 'config.ini')
    config = configparser.ConfigParser()
    config.read(configpath)
    class_detail = config.get('class_idx', 'class_detail')# 读取数据库配置
    dict_class_idx = json.loads(class_detail)

    # 获取tensor中前k个最大值及其索引
    r = torch.tensor(np.array(output))
    values, indices = torch.topk(r[0][0], k=5, dim=0, largest=True)
    print("Top 5 values:", values)
    print(softmax(values.detach().numpy()))
    print("Their indices:", indices)
    print([dict_class_idx.get(str(key), None) for key in indices.numpy()])

