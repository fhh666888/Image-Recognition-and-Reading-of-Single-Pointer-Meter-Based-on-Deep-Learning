# -*- coding:utf-8 -*-
import cv2
from math import *
import numpy as np
import time,math
import os
import re
#from CRAFT_5.test import test_to_cut


'''旋转图像并剪裁'''
def rotate( img, pt1, pt2, pt3, pt4 ):
    # print pt1,pt2,pt3,pt4
    withRect = math.sqrt((pt4[0] - pt1[0]) ** 2 + (pt4[1] - pt1[1]) ** 2)
    # 矩形框的宽度
    heightRect = math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) **2)
    # print withRect,heightRect
    angle = acos((pt4[0] - pt1[0]) / withRect) * (180 / math.pi)
    # 矩形框旋转角度 print angle
    # if pt3[1]>pt4[1]:
    #     print("顺时针旋转")
    # else:
    #     print("逆时针旋转")
    angle=angle - 90
    height = img.shape[0] # 原始图像高度
    width = img.shape[1]  # 原始图像宽度
    rotateMat = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1) # 按angle角度旋转图像
    heightNew = int(width * fabs(sin(radians(angle))) + height * fabs(cos(radians(angle))))
    widthNew = int(height * fabs(sin(radians(angle))) + width * fabs(cos(radians(angle))))
    rotateMat[0, 2] += (widthNew - width) / 2
    rotateMat[1, 2] += (heightNew - height) / 2
    imgRotation = cv2.warpAffine(img, rotateMat, (widthNew, heightNew), borderValue=(0, 0, 255))
    # cv2.imshow('rotateImg2', imgRotation)
    # cv2.waitKey(0)
        # 旋转后图像的四点坐标
    [[pt1[0]], [pt1[1]]] = np.dot(rotateMat, np.array([[pt1[0]], [pt1[1]], [1]]))
    [[pt3[0]], [pt3[1]]] = np.dot(rotateMat, np.array([[pt3[0]], [pt3[1]], [1]]))
    [[pt2[0]], [pt2[1]]] = np.dot(rotateMat, np.array([[pt2[0]], [pt2[1]], [1]]))
    [[pt4[0]], [pt4[1]]] = np.dot(rotateMat, np.array([[pt4[0]], [pt4[1]], [1]]))
    img_roi = imgRotation[int(float(pt1[1])):int(float(pt3[1])), int(float(pt1[0])):int(float(pt3[0]))]
    # cv2.imshow("img_roi", img_roi)  # 裁减得到的旋转矩形框
    # cv2.waitKey(0)
        # # 处理反转的情况
        # if pt2[1]>pt4[1]:
        #     pt2[1],pt4[1]=pt4[1],pt2[1]
        #     if pt1[0]>pt3[0]:
        #         pt1[0],pt3[0]=pt3[0],pt1[0]
        #         imgOut = imgRotation[int(pt2[1]):int(pt4[1]), int(pt1[0]):int(pt3[0])]
        #         cv2.imshow("imgOut", imgOut) # 裁减得到的旋转矩形框
        #         cv2.waitKey(0)
    return img_roi # rotated image ＃　根据四点画原矩形


def drawRect(img,pt1,pt2,pt3,pt4,color,lineWidth):
        # cv2.line(img, pt1, pt2, color, lineWidth)
        # cv2.line(img, pt2, pt3, color, lineWidth)
        # cv2.line(img, pt3, pt4, color, lineWidth)
        # cv2.line(img, pt1, pt4, color, lineWidth)
        pass

        #　读出文件中的坐标值
def ReadTxt(imageName,last):
        #fileTxt=directory+"//rawLabel//"+imageName[:7]+last # txt文件名
        getTxt=open(last, 'r') # 打开txt文件
        lines = getTxt.readlines()
        length=len(lines)

        pt1=list(map(float,lines[0].split(',')[0:2]))
        pt2=list(map(float,lines[0].split(',')[2:4]))
        pt3=list(map(float,lines[0].split(',')[4:6]))
        pt4=list(map(float,lines[0].split(',')[6:8])) # float转int
        pt2=list(map(int,pt2))
        pt1=list(map(int,pt1))
        pt4=list(map(int,pt4))
        pt3=list(map(int,pt3))
        imgSrc = cv2.imread(imageName)
        drawRect(imgSrc, tuple(pt1),tuple(pt2),tuple(pt3),tuple(pt4), (0, 255, 0), 2)
        # cv2.imshow("img", imgSrc)
        # cv2.waitKey(0)
        end = rotate(imgSrc,pt1,pt2,pt3,pt4)
        return end

#if __name__=="__main__":
def cut_to_crnn():
    # last = 'res_5.txt'
    # imageName="res_5.jpg"
    #test_to_cut()
    img_path = 'E:\\Papercode\\CRAFT_5\\imagedir'  # 图片路径
    label_path = 'E:\\Papercode\\CRAFT_5\\txtdir'  # txt文件路径
    save_path = 'E:\\Papercode\\cutrotate_6\\cutimage'    # 保存路径
    img_total = []
    label_total = []
    imgfile = os.listdir(img_path)
    labelfile = os.listdir(label_path)

    for filename in imgfile:
        name, type = os.path.splitext(filename)
        if type == ('.jpg' or '.png'):
            img_total.append(name)
    for filename in labelfile:
        name, type = os.path.splitext(filename)
        if type == '.txt':
            label_total.append(name)

    for _img in img_total:
        if _img in label_total:
            filename_img = _img + '.jpg'
            imageName = os.path.join(img_path, filename_img)
            filename_label = _img + '.txt'
            n = 1
            # 打开文件，编码格式'utf-8','r+'读写
            #ReadTxt(imageName, last)
            last = os.path.join(label_path, filename_label)
            filename_last = _img + "_" + str(n) + ".jpg"
            end = ReadTxt(imageName, last)
            cv2.imwrite(os.path.join(save_path, filename_last), end)
            n = n + 1
        else:
            continue