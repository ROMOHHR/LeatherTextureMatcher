#include <assert.h>
#include <vector>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

// 定义一个结构体来存储值和索引
struct ValueIndexPair {
    float value;
    int index;

    // 重载小于运算符，用于排序
    bool operator<(const ValueIndexPair& other) const {
        return value > other.value; // 为了降序排列
    }
};

std::vector<float> processImage(cv::Mat src){
    std::vector<cv::Mat> channels;
    cv::split(src, channels);
    for (auto &img : channels) {
        img = img.reshape(1, 1);
    }
    cv::Mat dst;
    cv::hconcat( channels, dst );
    std::vector<float> vec = dst.reshape(1, 1);
    std::vector<float> result;
    for(float f : vec){
        result.push_back(f/255);
    }
    return result;
}

std::vector<ValueIndexPair> onnxInfer(const char*, Ort::Value &, const char**, const char**, int);

#include <string>
int main(int argc, char* argv[]) {
    std::string imagepath = "../input_img/" + std::string(argv[1]) + "/0000.jpg";
    cv::Mat inputImage = cv::imread(imagepath.c_str());
    // cv::Mat inputImage = cv::imread("../input_img/Leather_texture_B_1/0000.jpg");
    if (inputImage.empty()) {
        std::cerr << "Failed to load image." << std::endl;
        return 1;
    }

    cv::Mat grayImage;
    // cv::cvtColor(inputImage, dst, cv::COLOR_BGR2RGB);
    cv::cvtColor(inputImage, grayImage, cv::COLOR_BGR2GRAY);

    cv::Mat resizedImage;
    cv::resize(grayImage, resizedImage, cv::Size(224,224));
    std::vector<float> input_image = processImage(resizedImage);

    
    std::array<int64_t, 4> input_shape{1, 1, 224, 224}; //mode input shape NCHW = 1x1xHxW
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_image.data(), input_image.size(), input_shape.data(), input_shape.size());

    const char* input_names[] = {"input"};
    const char* output_names[] = {"output"};
    std::vector<ValueIndexPair> k_largest;

    // 判断输入图片质量: {"0": "blurs", "1": "origin"}
    std::vector<ValueIndexPair> imageQuality = onnxInfer("../QualityAnalysis.onnx", input_tensor, input_names, output_names, 2);// 推理结果为2维向量
    k_largest.clear();
    k_largest.insert(k_largest.begin(), imageQuality.begin(), imageQuality.begin() + 2);// 输出最大的2个值及其索引
    if (k_largest.size() > 0 && k_largest[0].index != 1) {// index不为1，表示图片质量过低
        std::cerr << "Failed image quality." << std::endl;
        return 1;
    }

    // 进行皮革纹理匹配
    std::vector<ValueIndexPair> classIndices = onnxInfer("../TextureClassifierModel.onnx", input_tensor, input_names, output_names, 40);// 推理结果为40维向量
    k_largest.clear();
    k_largest.insert(k_largest.begin(), classIndices.begin(), classIndices.begin() + 6);// 输出最大的6个值及其索引
    for (const auto& pair : k_largest) {
        std::cout << "Value: " << pair.value << ", Index: " << pair.index << std::endl;
    }

    printf("Done!\n");
    return 0;
}

std::vector<ValueIndexPair> onnxInfer(const char* model_path, Ort::Value &input_tensor, const char* input_names[], const char* output_names[], int array_size)
{
	// 创建环境
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "Default");

    // 初始化SessionOptions
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    // session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    // 加载ONNX模型
    Ort::Session session(env, model_path, session_options);
    
    // 运行模型推理
    std::vector<Ort::Value> output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
    // 获取输出数据
    float* floatarr = output_tensors.front().GetTensorMutableData<float>();

	// // 索引向量(array_size为模型推理结果每个向量的维度数)
    // std::vector<int> result(array_size);
    // // 初始化索引
    // for (int i = 0; i < result.size(); ++i) result[i] = i;
	// // 对索引进行排序，基于数组中的值
    // std::sort(result.begin(), result.end(), [&](int i, int j) { return floatarr[i] < floatarr[j]; });
    

    // 创建一个包含值和索引对的向量
    std::vector<ValueIndexPair> pairs;
    for (int i = 0; i < array_size; ++i) {
        pairs.push_back({floatarr[i], i});
    }
    std::sort(pairs.begin(), pairs.end());

    return pairs;//result
}
