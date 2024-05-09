import cv2

image = cv2.imread('E:\\Papercode\\CRAFT_5\\tupian\\11.jpg')
print(image.shape)
cropped = image[300:450, 300:450]  # 裁剪坐标为[y0:y1, x0:x1]
cv2.imwrite('tupian/cropped.jpg', cropped)