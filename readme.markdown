**目录结构说明**：
- datasets: 数据集目录
    - srcImage: 原始数据集
    - Match: 皮革纹理匹配模型使用的数据集
    - filter: 判断输入图片质量模型的数据集
- models: 训练得到的模型目录
    - resnet18_c40_0.9889: 皮革纹理匹配模型文件目录
        - best_model.pth: 包含完整的PyTorch模型文件
        - best.pth: PyTorch模型参数文件
        - config.ini: 配置文件
    - resnet18_c2_0.8525: 判断输入图片质量模型文件目录
        - best_model.pth: 包含完整的PyTorch模型文件
        - best.pth: PyTorch模型参数文件
        - config.ini: 配置文件
- tools: 处理数据集的工具脚本目录
    - RandomCropImg.py: 在输入图片中随机截取指定大小、指定数量的图片
    - train_test_split.py: 按指定比例将训练数据划分成训练集和测试集数据
- deploy: 模型部署目录
    - onnx: 皮革纹理匹配模型onnx文件目录
        - pyTools: python脚本工具目录
            - pth2onnx.py: PyTorch文件转onnx文件脚本
            - onnx_predict.py: 使用onnx文件进行推理的脚本（验证onnx文件正确性）
            - onnx_infer.cpp: 使用C++和onnx文件进行部署及推理
        - test: 项目在Linux平台的测试代码目录
            - test.cpp: 检验onnx文件正确性
            - CMakeLists.txt
        - CMakeLists.txt: 使用onnx文件进行部署的cmake脚本
        - leather.cpp: 皮革纹理匹配模型API源码
        - TextureClassifierModel.onnx: 皮革纹理匹配模型文件
    - openvino: 皮革纹理匹配模型openvino文件目录
        - pyTools: python脚本工具目录
            - complie_mdoel.py: 检验openvino模型文件正确性
            - infer.py: 使用openvino模型文件进行推理
        - TextureClassifierModel.xml
        - TextureClassifierModel.bin
        - TextureClassifierModel.mapping
- TextureClassifier.py: 模型训练主脚本
- dataset_utils.py: 数据集加载器脚本
- optimizer.py: 优化器脚本
- metrics.py: 模型质量分析脚本
- predict.py: 模型预测验证脚本
- u20_leather_hhr.tar: 项目的docker镜像

---

# 皮革纹理匹配
## 项目需求
皮革厂生产大概2000种纹理的皮革，这些皮革种类都有样品和编号。
客户上门时，会带着他们的所需的皮革样品。这个时候，厂家需要将客户的样品跟库里的样品的纹理进行匹配，找出近似度最高的6种皮革，显示其编号及样品图。目前这个匹配的过程完全是靠人工，工作量很大，并且有些纹理非常接近，找出最近似的6种皮革也很困难。
因此，厂家要求开发一款产品，替代这个人工过程。有时候厂家的业务员会上门到客户那里去，这个时候客户的样品是在客户那里的，业务员通过产品可以拍摄样品图片或录制样品视频来进行匹配。
（备注：产品仅需匹配皮革的纹理，不需要考虑颜色，要求越方便越准确越好。）

## 方案设计
通过深度学习构建一个分类器模型，输入待匹配的样品图片或视频，给出置信度最高的前6个种类的编号。

- 数据集：通过实拍+网络上收集一定种类的皮革纹理图片。由于有的皮革纹理较细，有的纹理较粗，在进行分辨或训练时需要采用不同的比例尺，使得局部纹理与整体的纹理具有相似性。
收集到的皮革通常一张就是一个种类，有的可能是多张皮革的纹理相同或相似，仅仅是颜色不同。因此在处理数据集时，先是得到一整张皮革大分辨率的图片，之后根据纹理的粗细程度选择合适的比例尺，截取出若干分辨率较低的局部图片组成最终的数据集。

- 训练模型：由于需求较简单，训练数据也不需要太复杂，可对ImageNet相关的深度神经网络进行迁移学习来实现主要目标。


**产品初步流程为**：拍摄/接收用户提供的样品图片/视频、通过模型得到置信度最高的6个种类的编号、展示编号及样品图。

**优化**：
1. 考虑到提供给模型的图片可能存在模糊、高曝光、过暗等问题，会影响到最终的结果，因此可对传入的图片进行初步判断。
2. 需要对数据集进行预处理：彩色图转灰度图、统一尺寸。在进行模型推理时，也需要对输入图片进行同样的处理。