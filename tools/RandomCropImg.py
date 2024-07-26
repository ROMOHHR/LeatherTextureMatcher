from PIL import Image
import os
import random
import re
import shutil
 
''' 说明（简述：对指定图片进行随机裁剪，生成一组裁剪后的图片）
对指定目录下的所有图片进行随机裁剪，可指定裁剪区域的宽(w)、高(h)及要生成的裁剪图片个数(pic_num)；
每次生成裁剪图片时，可通过replace来控制对输出路径是覆盖(replace=True)或者追加(replace=False)
'''
 
def RandomCropImg_Default(input_path, output_path, h, w, pic_num, replace=True, crop_always=False):
    for each_image in os.listdir(input_path):
        image_input_fullname = os.path.join(input_path, each_image)
        if not os.path.isfile(image_input_fullname):
            continue

        image_name = each_image[: each_image.index('.')]  # 不带后缀的文件名
        image_format = each_image[each_image.index('.'):]

        # PIL库打开每一张图像
        img = Image.open(image_input_fullname)
        # 获取图片大小及中心点坐标
        x_max = img.size[0]
        y_max = img.size[1]

        if h > y_max or w > x_max:  # 要裁剪的尺寸比原始图像大，则不裁剪
            continue
        
        mid_point_x = int(x_max/2)
        mid_point_y = int(y_max/2)

        bFirst = True
        count = 0
        count_start = 0
        while count < pic_num:
            # 定义裁剪图片左上点和右下点的像素坐标
            ltop_x = mid_point_x + random.randint(-mid_point_x, mid_point_x-w)
            ltop_y = mid_point_y + random.randint(-mid_point_y, mid_point_y-h)
            rtop_x = ltop_x + w
            rtop_y = ltop_y + h

            if rtop_x > x_max or rtop_y > y_max:  # 裁剪区域超出原始图像范围，则不裁剪
                if crop_always:  # 如果超出范围也裁剪，则将原始图像右下点作为裁剪图像的右下点
                    rtop_x = x_max
                    rtop_y = y_max
                continue

            # 从原始图像返回一个矩形区域，区域是一个4元组定义左上右下像素坐标
            box = (ltop_x, ltop_y, rtop_x , rtop_y)
            # 进行roi裁剪
            roi_area = img.crop(box)
            # 裁剪后每个图像的路径+名称
            roi_area_path = os.path.join(output_path, image_name)

            if bFirst:
                if os.path.exists(roi_area_path):
                    if replace:
                        shutil.rmtree(roi_area_path)    #递归删除文件夹，即：删除非空文件夹
                    else:
                        names = os.listdir(roi_area_path)
                        if len(names) > 0:
                            names = list(filter(lambda x: re.match(r'^\d{4}[.]', x) != None, names))
                            names.sort()
                            count_start = int(names[-1][:4]) + 1
                bFirst = False
            
            os.makedirs(roi_area_path, exist_ok=True)  # 检查路径，不存在则创建

            roi_area_name = f"{count + count_start:04d}{image_format}"
            image_output_fullname = os.path.join(roi_area_path, roi_area_name)

            # 特殊处理
            roi_area = roi_area.resize((224, 224))

            # 存储裁剪得到的图像
            roi_area.save(image_output_fullname)
            count += 1

        print('{0} crop done.'.format(each_image))


if __name__ == '__main__':
    input_path = "./datasets/data3/input/S"
    output_path = "./datasets/data3/output2"
    h = 300
    w = 300
    pic_num = 5
    os.makedirs(output_path, exist_ok=True)  # 检查路径，不存在则创建
    RandomCropImg_Default(input_path, output_path, h, w, pic_num, replace=True, crop_always=False)
