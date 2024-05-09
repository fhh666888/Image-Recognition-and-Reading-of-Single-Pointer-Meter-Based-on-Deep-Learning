# -- coding: utf-8 --
import cv2
import os
from perspective_2.predict import to_crop


def update(input_img_path, output_img_path):
    image = cv2.imread(input_img_path)
    # print(image.shape)
    cropped = image[300:450, 300:450]  # 裁剪坐标为[y0:y1, x0:x1]
    cv2.imwrite(output_img_path, cropped)

def crop_to_CRAFT():
    #dataset_dir = 'D:\\detection\\yolov5-5.0\\runs\\333'  # 需裁剪的图片存放文件夹
    output_dir = 'E:\\Papercode\\perspective_2\\meter_distro\\cutimage'   # 裁好图片输出文件夹
    dataset_dir = to_crop('E:\\Papercode\\image\\')

    # 获得需要转化的图片路径并生成目标路径
    image_filenames = [(os.path.join(dataset_dir, x), os.path.join(output_dir, x))
                       for x in os.listdir(dataset_dir)]
    # 转化所有图片
    for path in image_filenames:
        update(path[0], path[1])
    return output_dir
