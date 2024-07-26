import os
import shutil
import random
import re
import cv2

def mkfile(file_path):
     if not os.path.exists(file_path):
            os.makedirs(file_path)

def train_test_split(file_path, split_rate=0.2):
    train_path = file_path + '/train/'
    test_path = file_path + '/test/'

    # 递归删除文件夹，即：删除非空文件夹
    if os.path.exists(train_path):
        shutil.rmtree(train_path)
    if os.path.exists(test_path):
        shutil.rmtree(test_path)

    # 获取data文件夹下所有文件夹名（即需要分类的类名）
    flower_class = os.listdir(file_path)
    flower_class = list(filter(lambda x: re.match(r'\D*\d+', x), flower_class))
    flower_class.sort(key=lambda x: int(re.sub(r'\D*(\d+)', r'\1', x)))

    # 创建 训练集train 和 测试集test 文件夹
    mkfile(train_path)
    mkfile(test_path)
    for cla in flower_class:
        mkfile(train_path + cla)
        mkfile(test_path + cla)

    # 遍历所有类别的全部图像并按比例分成训练集和验证集
    for cla in flower_class:
        cla_path = file_path + '/' + cla + '/'  # 某一类别的子目录
        images = os.listdir(cla_path)  # iamges 列表存储了该目录下所有图像的名称
        num = len(images)
        eval_index = random.sample(images, k=int(num * split_rate))  # 从images列表中随机抽取 k 个图像名称
        for index, image in enumerate(images):
            # eval_index 中保存验证集val的图像名称
            if image in eval_index:
                image_path = cla_path + image
                new_path = test_path + cla
                shutil.copy(image_path, new_path)  # 将选中的图像复制到新路径

            # 其余的图像保存在训练集train中
            else:
                image_path = cla_path + image
                new_path = train_path + cla
                shutil.copy(image_path, new_path)
            print("\r[{}] processing [{}/{}]".format(cla, index + 1, num), end="")  # processing bar
        print()

    print("processing done!")


def image_blurs(file_path):
    origin_path = file_path + '/origin/'
    output_path = file_path + '/blurs/'

    # 递归删除文件夹，即：删除非空文件夹
    if os.path.exists(origin_path):
        shutil.rmtree(origin_path)
    if os.path.exists(output_path):
        shutil.rmtree(output_path)

    # 获取data文件夹下所有文件夹名
    flower_class = os.listdir(file_path)
    flower_class = list(filter(lambda x: re.match(r'\D*\d+', x), flower_class))
    flower_class.sort(key=lambda x: int(re.sub(r'\D*(\d+)', r'\1', x)))

    # 创建 训练集train 和 测试集test 文件夹
    mkfile(origin_path)
    mkfile(output_path)

    for cla in flower_class:
        cla_path = file_path + '/' + cla + '/'  # 某一类别的子目录
        images = os.listdir(cla_path)  # iamges 列表存储了该目录下所有图像的名称
        num = len(images)
        split_rate = 0.8  # 不进行处理的图片比例
        eval_index = random.sample(images, k=int(num * split_rate))  # 从images列表中随机抽取 k 个图像名称
        for index, image in enumerate(images):
            imgName = image[:image.find(".")]
            if image in eval_index:
                image_path = cla_path + image
                outorigin_path = origin_path + cla + "_" + imgName + "_origin.jpg"
                shutil.copy(image_path, outorigin_path)  # 将选中的图像复制到新路径
            else:
                cvimage = cv2.imread(cla_path + image)
                height, width = cvimage.shape[:2]
                sigma_percentage1 = random.uniform(0.08, 0.2)
                # sigma_percentage2 = random.uniform(0.04, 0.08)
                sigma_percentage3 = random.uniform(0.06, 0.1)

                sigma = int(sigma_percentage1 * min(height, width))
                sigma = sigma+1 if sigma%2==0 else sigma
                # 高斯模糊
                gaussian_blurred = cv2.GaussianBlur(cvimage, (sigma,sigma), 0)

                # sigma = int(sigma_percentage2 * min(height, width))
                # sigma = sigma+1 if sigma%2==0 else sigma
                # # 高斯模糊
                # gaussian_blurred2 = cv2.GaussianBlur(cvimage, (sigma,sigma), 0)
                
                sigma = int(sigma_percentage3 * min(height, width))
                sigma = sigma+1 if sigma%2==0 else sigma
                # 均值模糊
                mean_blurred = cv2.blur(cvimage, (sigma,sigma))
                # 中值模糊
                median_blurred = cv2.medianBlur(cvimage, sigma)
                
                cv2.imwrite(output_path + cla + "_" + imgName + "_gaussian_blurred.jpg", gaussian_blurred)
                # cv2.imwrite(output_path + imgName + "_gaussian_blurred2.jpg", gaussian_blurred2)
                cv2.imwrite(output_path + cla + "_" + imgName + "_mean_blurred.jpg", mean_blurred)
                cv2.imwrite(output_path + cla + "_" + imgName + "_median_blurred.jpg", median_blurred)

            print("\r[{}] processing [{}/{}]".format(cla, index + 1, num), end="")  # processing bar
        print()

    print("processing done!")


if __name__ == '__main__':
    file_path = './datasets/data3/output'
    # file_path = './datasets/data4'
    if 0:
        file_path = './datasets/data3/output/test'
        image_blurs(file_path)
    else:
        split_rate = 0.3  # (测试集比例) 划分比例，训练集 : 测试集 = 8 : 2
        train_test_split(file_path, split_rate=split_rate)
