import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

"""
        ！！！   opencv读取的图片格式为BGR格式
"""

"""
        彩色图（RGB）转灰度图（GRAY）
        公式 ： GRAY = （R * 0.3 + G * 0.59 + B * 0.11）
"""
# 使用Opencv转彩色图转灰度图
img = cv2.imread(r'G:\FPS-AI\demo for img\640.png')  # opencv读取图片
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 使用PIL彩色图转灰度图
# img0 = Image.open(r'G:\FPS-AI\demo for img\640.png', mode='r')  # PIL读取图片
# img_gray = img0.convert("L")  # 以灰度图像读取
# img_gray.show()

# 手动转灰度图
# img = cv2.imread(r'G:\FPS-AI\demo for img\640.png')  # opencv读取图片
# h, w = img.shape[:2]  # 获取图片的h和w————————shape（）返回(h, w, c)， h和w均为像素点个数， h高度，w宽度，c通道数
# gray = np.zeros([h, w], img.dtype)  # 创建一张和当前图片同尺寸的单通道图片，拿来记录灰度值
# for i in range(h):
#     for j in range(w):
#         BGR_val = img[i, j]  # 获取当前坐标的BGR值，依次往下，第一层B，第二层G，第三层R
#         gray[i, j] = int(BGR_val[0] * 0.11 + BGR_val[1] * 0.59 + BGR_val[2] * 0.3)  # BGR_val = [B , G , R]


"""
        二值化
        参考  ： https://zhuanlan.zhihu.com/p/186357948
        参考  ： https://blog.csdn.net/weixin_46192930/article/details/108607453
        在灰度图的基础上，剔除图像内像数值高于或低于一定值的像素点
        threshold(gray_src, dst, threshold_value, threshold_max,THRESH_BINARY);

        OpenCV中图像二值化方法：   cv2.threshold(gray, 127, 255, cv2.THRESH_TOZERO)      //原图，目标图，已知阈值，阈值最大值，阈值类型
        返回阈值的值和图像。
        
        THRESH_BINARY           阈值二值化
        THRESH_BINARY_INV       阈值反二值化
        THRESH_TRUNC            截断 
        THRESH_TOZERO           阈值取零 
        THRESH_TOZERO_INV       阈值反取零 
        
        取阈值方法：  cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)    //原图，目标图，已知阈值，阈值最大值，二值化方法 | 阈值类型
        
        THRESH_TRIANGLE         三角二值化
        THRESH_OTSU             OTSU阈值法
        
        自适应阈值处理可以使用变换的阈值。它通过计算每个像素点周围邻近区域的加权平均值获得阈值，然后处理。这个方式可以更好的处理明暗差异较大的图像
        dst = cv.adaptiveThreshold( src, maxValue, adaptiveMethod, thresholdType, blockSize, C )
        ADAPTIVE_THRESH_MEAN_C
        ADAPTIVE_THRESH_GAUSSIAN_C
        
        中值滤波不存在均值滤波等方式带来的细节模糊问题，可以几乎不影响原图的情况下去除噪声，但运算量大。
        gray = cv2.medianBlur(img,3)
        
        自定义分割,  将数组矩阵拉直然后求出平均数
        1、获取h和w
        2、
        
"""
# 将数组矩阵拉直，求均值阈值
h, w = gray.shape[:2]
m = np.reshape(gray, [1, w * h])  # 使用reshape( )方法来更改数组的形状
mean = m.sum() / (w * h)

# 手动二值化
rows, cols = gray.shape
for i in range(rows):
    for j in range(cols):
        if gray[i, j] <= mean:
            gray[i, j] = 0
        else:
            gray[i, j] = 255

# 使用Opencv二值化
# ret, gray = cv2.threshold(gray, mean, 255, cv2.THRESH_BINARY)
# ret, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# ret = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,5,3)
print('当前阈值 ： ', mean)

while True:
    cv2.imshow("img for gray", gray)  # 显示灰度图
    if cv2.waitKey(1) & 0xff == ord('q'):  # 按q退出
        cv2.destroyAllWindows()  # 如果之前没有释放掉内存的操作的话destroyallWIndows会释放掉被那个变量占用的内存
        break
