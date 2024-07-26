#include<stdio.h>
#include<stdlib.h>
#include<time.h>

#include <algorithm>
#include <vector>
#include <onnxruntime_cxx_api.h>


std::vector<int> onnxInfer(const char*, Ort::Value &, const char**, const char**, int);

void processImageData(unsigned char * imageData, int * result)
{
    // 在调用本函数处保证imageData是 1*3*224*224 的图片数据
    // 处理输入图片数据
    int width = 224;
    int height = 224;
    // 获取模型输入的维度信息
    std::vector<int64_t> input_node_dims = {1, 1, 224, 224};
	// 创建一个向量来存储转换后的浮点数数据
    std::vector<float> input_data(1 * 1 * 224 * 224);
    // 将unsigned char数据转换为浮点数
    // for (int i = 0; i < 1 * 1 * 224 * 224; ++i) {
    //     // 这里我们假设像素值在0-255之间，并且我们希望将其归一化到0-1范围
    //     input_data[i] = static_cast<float>(imageData[i]) / 255.0f;
    // }
    for (int y = 0; y < height; ++y) { //输入数据为灰度图
        for (int x = 0; x < width; ++x) {
            int idx = (y * width + x) * 3; // Index for the RGB pixel
            float r = static_cast<float>(imageData[idx]);
            float g = static_cast<float>(imageData[idx + 1]);
            float b = static_cast<float>(imageData[idx + 2]);
            
            // Convert to grayscale and store
            input_data[y * width + x] = (0.299f * r + 0.587f * g + 0.114f * b) / 255.0f;
        }
    }
    const char* input_names[] = {"input"};
    const char* output_names[] = {"output"};

    // 创建输入Tensor
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_data.data(), input_data.size(), input_node_dims.data(), input_node_dims.size());


    // 判断输入图片质量: {"0": "blurs", "1": "origin"}
    std::vector<int> imageQuality = onnxInfer("QualityAnalysis.onnx", input_tensor, input_names, output_names, 2);//推理结果为2维向量
    if (imageQuality.size() > 0 && imageQuality[0] != 1) return; // 输入图片不合格


    std::vector<int> classIndices = onnxInfer("TextureClassifierModel.onnx", input_tensor, input_names, output_names, 40);//推理结果为40维向量
	for (int i = 0; i < 6; ++i) {// 保存前6个最大值的索引号
        result[i] = classIndices[i];
    }
}

std::vector<int> onnxInfer(const char* model_path, Ort::Value &input_tensor, const char* input_names[], const char* output_names[], int array_size)
{
	// 创建环境
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "Default");

    // 初始化SessionOptions
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);

    // 加载ONNX模型
    Ort::Session session(env, model_path, session_options);
    
    // 运行模型推理
    std::vector<Ort::Value> output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
    // 获取输出数据
    float* floatarr = output_tensors.front().GetTensorMutableData<float>();

	// 索引向量(array_size为模型推理结果每个向量的维度数)
    std::vector<int> indices(array_size);
    // 初始化索引
    for (int i = 0; i < indices.size(); ++i) indices[i] = i;
	// 对索引进行排序，基于数组中的值（倒序排列）
    std::sort(indices.begin(), indices.end(), [&](int i, int j) { return floatarr[i] > floatarr[j]; });
    
    return indices;
}
