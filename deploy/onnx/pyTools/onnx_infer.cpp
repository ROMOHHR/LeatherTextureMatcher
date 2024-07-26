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

void softmax(float* input, int size) {
    float maxVal = -1e9;
    for (int i = 0; i < size; ++i) {
        if (input[i] > maxVal) {
            maxVal = input[i];
        }
    }
    float sumExp = 0.0;
    for (int j = 0; j < size; ++j) {
        input[j] = exp(input[j] - maxVal);
        sumExp += input[j];
    }
    for (int k = 0; k < size; ++k) {
        input[k] /= sumExp;
    }
}

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
#include <string>
int main(int argc, char* argv[]) {
    cv::Mat inputImage;
    if(argc > 1){
        std::string str = "../input_img/"   + std::string(argv[1]) + "/0000.jpg";
        inputImage = cv::imread(str.c_str());
    }
    else{
        inputImage = cv::imread("../input_img/Leather_texture_B_1/0000.jpg");
    }

    if (inputImage.empty()) {
        std::cerr << "Failed to load image." << std::endl;
        return 1;
    }

    cv::Mat dst;
    // cv::cvtColor(inputImage, dst, cv::COLOR_BGR2RGB);
    cv::cvtColor(inputImage, dst, cv::COLOR_RGB2GRAY);

    cv::Mat resizedImage;
    cv::resize(dst, resizedImage, cv::Size(224,224));
    std::vector<float> input_image_ = processImage(resizedImage);

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "Default");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);

    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    const char* model_path = "../resnet18_c40_0.9889/TextureClassifierModel.onnx";

    Ort::Session session_(env, model_path, session_options);

    static constexpr const int height_ = 224; //model input height
    static constexpr const int width_ = 224; //model input width
    Ort::Value input_tensor_{nullptr};
    std::array<int64_t, 4> input_shape_{1, 1, height_, width_}; //mode input shape NCHW = 1x1xHxW
    Ort::Value output_tensor_{nullptr};
    std::array<int64_t, 2> output_shape_{1, 40}; //model output shape,
    std::array<_Float32, 40> results_{};

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    input_tensor_ = Ort::Value::CreateTensor<float>(memory_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());
    output_tensor_ = Ort::Value::CreateTensor<float>(memory_info, results_.data(), results_.size(), output_shape_.data(), output_shape_.size());
    const char* input_names[] = {"input"};
    const char* output_names[] = {"output"};
    session_.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor_, 1, output_names, &output_tensor_, 1);
    float* out = output_tensor_.GetTensorMutableData<float>();


    // std::cout<<out[0]<<":"<<out[1]<<std::endl;
    int size = 40; // 40个类的置信度
    int k = 5; // 前k个最大值
    softmax(out, 40);

    // 创建一个包含值和索引对的向量
    std::vector<ValueIndexPair> pairs;
    for (int i = 0; i < size; ++i) {
        pairs.push_back({out[i], i});
    }

    // 对向量进行排序，以找到最大的k个值
    std::sort(pairs.begin(), pairs.end());
    std::vector<ValueIndexPair> k_largest;

    // 提取最大的k个值及其索引
    k_largest.clear();
    k_largest.insert(k_largest.begin(), pairs.begin(), pairs.begin() + k);

    // 输出最大的k个值及其索引
    for (const auto& pair : k_largest) {
        std::cout << "Value: " << pair.value << ", Index: " << pair.index << std::endl;
    }

    printf("Done!\n");
    return 0;
}