import os
import cv2
import json
import numpy as np
import configparser
from openvino.runtime import Core

def softmax(x):
    e_x = np.exp(x)
    return e_x / e_x.sum(axis=0)

# 创建推理请求
ie = Core()
model_xml = "openvino/TextureClassifierModel.xml"
model = ie.read_model(model=model_xml)
compiled_model = ie.compile_model(model=model, device_name="CPU")
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

# 设置输入
image_filename = r"datasets\data3\output\Leather_texture_S_22\0000.jpg"
image = cv2.imread(image_filename)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# N,C,H,W = batch size, number of channels, height, width.
# 由于 input_layer.shape 是动态的，一张图片直接用固定的形状
N, C, H, W = 1, 1, 224, 224

# OpenCV resize expects the destination size as (width, height).
resized_image = cv2.resize(src=image, dsize=(W, H))
resized_image = resized_image.astype(np.float32)/255.0
# 由于RGB转灰度图，通道维度变成1，但是被opencv优化去掉了，需要扩展2次维度（batch_size, channel）
input_data = np.expand_dims(resized_image, 0).astype(np.float32)
input_data = np.expand_dims(input_data, 0).astype(np.float32)
# RGB三通道则要把通道数维度放到前面后再扩展一个维度（batch_size）
# input_data = np.expand_dims(np.transpose(resized_image, (2, 0, 1)), 0).astype(np.float32)

# 执行推理
# for single input models only
result = compiled_model(input_data)[output_layer]
# # for multiple inputs in a list
# result = compiled_model([input_data])[output_layer]
# # or using a dictionary, where the key is input tensor name or index
# result = compiled_model({input_layer.any_name: input_data})[output_layer]

# 获取分类信息（分类数量、索引对应的种类）
model_dir = os.path.join('models', 'resnet50_c40_0.9833')
configpath = os.path.join(model_dir, 'config.ini')
config = configparser.ConfigParser()
config.read(configpath)
class_detail = config.get('class_idx', 'class_detail')# 读取数据库配置
dict_class_idx = json.loads(class_detail)

# 查看推理结果
k = 5
res = softmax(result[0])
indices = np.argsort(-res)
top_k_indices = indices[:k]
top_k_values = np.take(res, top_k_indices)
print("Top {} largest values:".format(k))
for value, index in zip(top_k_values, top_k_indices):
    print("Value: {:.4f}, Index: {}, class: {}".format(value, index, dict_class_idx[str(index)]))
