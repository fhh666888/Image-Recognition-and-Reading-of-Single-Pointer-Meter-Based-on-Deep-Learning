import math
import os
import cv2
import numpy
import torch
from models.net import U2NET
from models.attnet import AttU2Net
from models.attentionU2NetPro import AttU2NetPro
from crnn.demo import demo_to_read
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
#from models.AttentionGate1 import u2net_full
from attention_block.AttentionGate1 import u2net_full


class MeterReader(object):

    def __init__(self, is_cuda=False):
        #self.net = U2NET(3, 2)
        self.net = u2net_full()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() and is_cuda else 'cpu')
        self.net.load_state_dict(torch.load('E:\\Papercode\\ReadMeter_7\\weight\\512.pt', map_location='cpu'))
        self.net.eval().to(self.device)
        '''以下为超参数，需要根据不同表盘类型设定'''
        self.line_width = 1600  # 表盘展开为直线的长度,按照表盘的图像像素周长计算
        self.line_height = 150  # 表盘展开为直线的宽度，该设定按照围绕圆心的扇形宽度计算，需要保证包含刻度以及一部分指针
        self.circle_radius = 205  # 预设圆盘直径，扫描的最大直径，需要小于图幅，否者可能会报错
        self.circle_center = [207, 207]  # 圆盘指针的旋转中心，预设的指针旋转中心
        self.pi = 3.1415926535898  # 参数PI
        self.threshold = 0.5  # 分割的阈值

    @torch.no_grad()
    def __call__(self, image):
        image = self.square_picture(image, 416)   # 将图片尺寸变换为416*416
        image_tensor = self.to_tensor(image.copy()).to(self.device)
        #d0, d1, d2, d3, d4, d5, d6 = self.net(image_tensor)
        d0 = self.net(image_tensor)
        mask = d0.squeeze(0).cpu().numpy()
        point_mask = self.binary_image(mask[0])
        dail_mask = self.binary_image(mask[1])

        kernel = numpy.ones((3, 3), numpy.uint8)
        kernel1 = numpy.ones((2, 2), numpy.uint8)
        kernel2 = numpy.ones((5, 5), numpy.uint8)
        point_mask = cv2.erode(point_mask, kernel2,5)
        point_mask = cv2.dilate(point_mask, kernel)
        dail_mask = cv2.erode(dail_mask, kernel)
        dail_mask = cv2.dilate(dail_mask, kernel1)

        # for i in range(95, 100):
        #     for j in range(310, 315):
        #         dail_mask[j][i] = 1
        relative_value = self.get_relative_value(point_mask, dail_mask)
        print(relative_value)


        """绘制分割结果图"""
        cv2.imshow('point_mask', point_mask)
        cv2.imshow('dail_mask', dail_mask)
        cv2.waitKey()
        # cv2.imshow('image', image)
        condition = dail_mask == 1
        image[condition] = (0, 255, 0)
        condition = point_mask == 1
        image[condition] = (0,0,255)

        # cv2.imshow('image_mask', image)
        # cv2.waitKey()

        return relative_value['ratio'],image

    def binary_image(self, image):
        """图片二值化"""
        condition = image > self.threshold
        image[condition] = 1
        image[~condition] = 0
        image = self.corrosion(image)
        return image

    def get_relative_value(self, image_pointer, image_dail):
        '''表盘图片展平'''
        line_image_pointer = self.create_line_image(image_pointer)
        line_image_dail = self.create_line_image(image_dail)
        '''二维图像转换为1为数组'''
        data_1d_pointer = self.convert_1d_data(line_image_pointer)
        data_1d_dail = self.convert_1d_data(line_image_dail)
        data_1d_dail = self.mean_filtration(data_1d_dail)  # 均值滤波

        """绘制一维数组结果图"""
        # plt.plot(numpy.arange(0, len(data_1d_pointer)), data_1d_pointer)
        # plt.plot(numpy.arange(0, len(data_1d_dail)), data_1d_dail)
        # plt.show()
        # exit()

        # cv2.imshow('line_image_pointer', line_image_pointer)
        # cv2.imshow('line_image_dail', line_image_dail)
        # cv2.waitKey()
        '''定位指针相对刻度位置'''
        dail_flag = False
        pointer_flag = False
        one_dail_start = 0
        one_dail_end = 0
        one_pointer_start = 0
        one_pointer_end = 0
        dail_location = []
        pointer_location = 0
        for i in range(self.line_width - 1):
            if data_1d_dail[i] > 0 and data_1d_dail[i + 1] > 0:
                if not dail_flag:
                    one_dail_start = i
                    dail_flag = True
            if dail_flag:
                if data_1d_dail[i] == 0 and data_1d_dail[i + 1] == 0:
                    one_dail_end = i - 1
                    one_dail_location = (one_dail_start + one_dail_end) / 2
                    dail_location.append(one_dail_location)
                    one_dail_start = 0
                    one_dail_end = 0
                    dail_flag = False
            if data_1d_pointer[i] > 0 and data_1d_dail[i + 1] > 0:
                if not pointer_flag:
                    one_pointer_start = i
                    pointer_flag = True
            if pointer_flag:
                if data_1d_pointer[i] == 0 and data_1d_pointer[i + 1] == 0:
                    one_pointer_end = i - 1
                    pointer_location = (one_pointer_start + one_pointer_end) / 2
                    one_pointer_start = 0
                    one_pointer_end = 0
                    pointer_flag = False
        scale_num = len(dail_location)
        num_scale = -1
        ratio = -1
        if scale_num > 0:
            for i in range(scale_num - 1):
                if dail_location[i] <= pointer_location < dail_location[i + 1]:
                    num_scale = i + (pointer_location - dail_location[i]) / (
                            dail_location[i + 1] - dail_location[i] + 1e-5) + 1
            ratio = (pointer_location - dail_location[0]) / (dail_location[-1] - dail_location[0] + 1e-5)
        result = {'scale_num': scale_num, 'num_sacle': num_scale, 'ratio': ratio}
        return result

    def create_line_image(self, image_mask):
        """
        创建线性图
        :param image_mask: 掩码图
        :return:
        """
        line_image = numpy.zeros((self.line_height, self.line_width), dtype=numpy.uint8)
        for row in range(self.line_height):
            for col in range(self.line_width):
                """计算与-y轴的夹角"""
                theta = ((2 * self.pi) / self.line_width) * (col + 1)
                '''计算当前扫描点位对应于原图的直径'''
                radius = self.circle_radius - row - 1
                '''计算当前扫描点对应于原图的位置'''
                y = int(self.circle_center[0] + radius * math.cos(theta) + 0.5)
                x = int(self.circle_center[1] - radius * math.sin(theta) + 0.5)
                line_image[row, col] = image_mask[y, x]
        plt.imshow(line_image)
        plt.show()
        return line_image

    def convert_1d_data(self, line_image):
        """
        将图片转换为1维数组
        :param line_image: 展开的图片
        :return: 一维数组
        """
        data_1d = numpy.zeros((self.line_width), dtype=numpy.int16)
        for col in range(self.line_width):
            for row in range(self.line_height):
                if line_image[row, col] == 1:
                    data_1d[col] += 1
        return data_1d

    def corrosion(self, image):
        """
        腐蚀操作
        :param image:
        :return:
        """
        kernel = numpy.ones((3, 3), numpy.uint8)
        image = cv2.erode(image, kernel)
        return image

    def mean_filtration(self, data_1d_dail):
        """
        均值滤波
        :param data_1d_dail:
        :return:
        """
        mean_data = numpy.mean(data_1d_dail)
        for col in range(self.line_width):
            if data_1d_dail[col] < mean_data:
                data_1d_dail[col] = 0
        return data_1d_dail

    @staticmethod
    def to_tensor(image):
        image = torch.tensor(image).float() / 255
        image = image.permute(2, 0, 1).unsqueeze(0)
        return image

    @staticmethod
    def square_picture(image, image_size):
        """
        任意图片正方形中心化
        :param image: 图片
        :param image_size: 输出图片的尺寸
        :return: 输出图片
        """
        h1, w1, _ = image.shape
        max_len = max(h1, w1)
        #max_len = 512
        fx = image_size / max_len
        fy = image_size / max_len
        image = cv2.resize(image, None, fx=fx, fy=fy, interpolation=cv2.INTER_AREA)
        h2, w2, _ = image.shape
        background = numpy.zeros((image_size, image_size, 3), dtype=numpy.uint8)
        background[:, :, :] = 127
        s_h = image_size // 2 - h2 // 2
        s_w = image_size // 2 - w2 // 2
        background[s_h:s_h + h2, s_w:s_w + w2] = image
        return background


# 修改图像的亮度，brightness取值0～2 <1表示变暗 >1表示变亮
def change_brightness(img, brightness):
    [averB, averG, averR] = numpy.array(cv2.mean(img))[:-1] / 3
    k = numpy.ones((img.shape))
    k[:, :, 0] *= averB
    k[:, :, 1] *= averG
    k[:, :, 2] *= averR
    img = img + (brightness - 1) * k
    img[img > 255] = 255
    img[img < 0] = 0
    return img.astype(numpy.uint8)


# 修改图像的对比度,coefficent>0, <1降低对比度,>1提升对比度 建议0-2
def change_contrast(img, coefficent):
    imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    m = cv2.mean(img)[0]
    graynew = m + coefficent * (imggray - m)
    img1 = numpy.zeros(img.shape, numpy.float32)
    k = numpy.divide(graynew, imggray, out=numpy.zeros_like(graynew), where=imggray != 0)
    img1[:, :, 0] = img[:, :, 0] * k
    img1[:, :, 1] = img[:, :, 1] * k
    img1[:, :, 2] = img[:, :, 2] * k
    img1[img1 > 255] = 255
    img1[img1 < 0] = 0
    return img1.astype(numpy.uint8)


if __name__ == '__main__':
    tester = MeterReader()
    #root = 'E:\\Papercode\\perspective_2\\meter_distro\\newimage'
    root = 'E:\\Papercode\\image'
    for image_name in os.listdir(root):
        path = f'{root}/{image_name}'
        image = cv2.imread(path)
        # image1 = change_brightness(image, 0.9)
        # image = change_contrast(image1, 1.8)
        i = image_name.split('.')[0]
        result,image = tester(image)
        result = float(result)
        #max = float(demo_to_read('E:\\Papercode\\cutrotate_6\\cutimage\\res_{}_1.jpg'.format(i)))
        max=1.6
        print(result*max)
        cv2.putText(image, str(round(float(result*max),3)), (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2,
                    cv2.LINE_AA)
        cv2.imwrite('E:\\Papercode\\result\\result_{}.jpg'.format(i), image)
        #show_img1('E:\\Papercode\\result\\result_a_1.jpg')
        # cv2.imshow('a', image)
        # cv2.waitKey()
