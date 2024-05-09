import os
import cv2
from perspective_2.meter_distro.ditro import MeterReader_distro

# parse arguments
def to_crop(predict_dir):
    #predict_dir='E:\\train\\'
    meter_distro = MeterReader_distro()
    image_list = os.listdir(predict_dir)
    save_path = 'E:\\Papercode\\perspective_2\\meter_distro\\newimage\\'

    for i in image_list:
        filename_last = str(i)
        print("**********", i)
        image = cv2.imread(predict_dir + i)
        restro_image, circle_center = meter_distro(image, i)   # 矫正图和圆心
        cv2.imwrite(os.path.join(save_path, filename_last), restro_image)
    return save_path


