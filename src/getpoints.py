import cv2  # 导入OpenCV库
import numpy as np  # 导入NumPy库

def get_points(left_image, right_image, area_threshold=1000, display=False):
    """
    从左右图像中提取质心点并可视化。
    :param left_image: 左图像，灰度图像格式
    :param right_image: 右图像，灰度图像格式
    :param area_threshold: 面积阈值，用于过滤掉面积大于该值的轮廓，默认值为1000
    :param display: 是否显示质心点，默认为False
    """
    # 二值化
    _, left_binary = cv2.threshold(left_image, 200, 255, cv2.THRESH_BINARY)  # 对左图像进行二值化处理
    _, right_binary = cv2.threshold(right_image, 200, 255, cv2.THRESH_BINARY)  # 对右图像进行二值化处理

    # 轮廓检测
    left_contours, _ = cv2.findContours(left_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 检测左图像的轮廓
    right_contours, _ = cv2.findContours(right_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 检测右图像的轮廓

    # 计算质心
    def get_centers(contours, area_threshold):
        """
        计算轮廓的质心。
        :param contours: 轮廓列表，每个轮廓是一个点的集合
        :param area_threshold: 面积阈值，用于过滤掉面积大于该值的轮廓
        :return: 返回质心坐标列表，每个质心是一个 (x, y) 元组
        """
        centers = []  # 初始化质心列表
        for contour in contours:  # 遍历所有轮廓
            area = cv2.contourArea(contour)  # 计算轮廓面积
            if area < area_threshold:  # 过滤掉面积大于阈值的轮廓
                M = cv2.moments(contour)  # 计算轮廓的矩
                if M["m00"] != 0:  # 确保分母不为零
                    cX = int(M["m10"] / M["m00"])  # 计算质心的X坐标
                    cY = int(M["m01"] / M["m00"])  # 计算质心的Y坐标
                    centers.append((cX, cY))  # 将质心坐标添加到列表中
        return centers  # 返回质心列表

    left_centers_image = np.zeros_like(left_image)  # 创建一个与左图像大小相同的空白图像
    right_centers_image = np.zeros_like(right_image)  # 创建一个与右图像大小相同的空白图像
    left_centers = get_centers(left_contours, area_threshold)  # 获取左图像的质心
    right_centers = get_centers(right_contours, area_threshold)  # 获取右图像的质心

    # 绘制质心
    if display:  # 如果display参数为True
        for left_center, right_center in zip(left_centers, right_centers):  # 遍历左右图像的质心
            # print(left_center)  # 打印左图像的质心坐标
            left_centers_image = cv2.circle(left_centers_image, left_center, 20, (255, 255, 255), -1)  # 在左图像上绘制质心
            right_centers_image = cv2.circle(right_centers_image, right_center, 20, (255, 255, 255), -1)  # 在右图像上绘制质心
        cv2.imshow("Left Points", left_centers_image)  # 显示左图像的质心
        cv2.imshow("Right Points", right_centers_image)  # 显示右图像的质心
        cv2.waitKey(0)  # 等待按键
        cv2.destroyAllWindows()  # 关闭所有窗口

if __name__ == "__main__":
    left_image = cv2.imread("./0.JPG", cv2.IMREAD_GRAYSCALE)  # 读取左图像
    right_image = cv2.imread("./0.JPG", cv2.IMREAD_GRAYSCALE)  # 读取右图像
    get_points(left_image, right_image, display=True)  # 调用get_points函数并显示结果