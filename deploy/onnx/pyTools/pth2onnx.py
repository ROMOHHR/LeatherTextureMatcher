import torch
import torch.onnx

# 假设 'model.pth' 是你的 PyTorch 模型文件
model_path = r'models\resnet18_c40_0.9889\best_model.pth'
model_path = r'models\resnet18_c2_0.8525\best_model.pth'

# 加载模型
# 加载你的模型，例如 model = TheModelClass(*args, **kwargs)
model = torch.load(model_path)
model.eval()  # 确保模型处于评估模式

# 创建一个符合模型输入尺寸的随机张量
# 假设模型输入尺寸为 (batch_size, channels, height, width)
batch_size = 1  # 例如
input_shape = (1, 224, 224)  # 例如
x = torch.randn(batch_size, *input_shape)

# 导出模型
output_onnx = 'TextureClassifierModel.onnx'
output_onnx = 'QualityAnalysis.onnx'
torch.onnx.export(model,               # 模型
                  x,                    # 模型输入 (或一个变量，如果模型有多个输入)
                  output_onnx,          # 输出 ONNX 文件的名称
                  export_params=True,   # 如果需要导出参数
                  opset_version=10,     # ONNX 算子集版本
                  do_constant_folding=True,  # 是否执行常量折叠优化
                  input_names=['input'],   # 输入名
                  output_names=['output'], # 输出名
                  dynamic_axes={'input': {0: 'batch_size'},  # 动态轴
                                'output': {0: 'batch_size'}})

print("模型已导出为 ONNX 格式: ", output_onnx)
