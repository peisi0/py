import cv2
import numpy as np


image = cv2.imread('image.jpg', 0)

if image is None:
    print("错误：未找到图片，请检查路径。")
else:
    # 2. 高斯模糊 (去噪)
    # (5, 5) 是核的大小，必须是奇数。数值越大，模糊越厉害，噪点越少，但细节也越少。
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # 3. 方法 A: 使用 Canny 边缘检测 (通常比固定阈值更准)
    # 30 和 150 是最小和最大阈值，可以根据图片对比度调整
    edges = cv2.Canny(blurred, 30, 150)

    # --- or ---

    # adaptive_thresh = cv2.adaptiveThreshold(blurred, 255, 
    #                                         cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
    #                                         cv2.THRESH_BINARY, 11, 2)

    # 4. 形态学操作 (闭运算)
    # 用来连接断开的轮廓，填补小孔
    kernel = np.ones((3, 3), np.uint8)
    # Close = 先膨胀后腐蚀
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # 5. 寻找轮廓
    contours, hierarchy = cv2.findContours(closed_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 6. 绘制结果
    output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(output, contours, -1, (0, 255, 0), 2)

    cv2.namedWindow('Original', cv2.WINDOW_NORMAL) 
    cv2.namedWindow('Edges (Processed)', cv2.WINDOW_NORMAL) 
    cv2.namedWindow('Final Contours', cv2.WINDOW_NORMAL) 

    # 显示对比
    cv2.imshow('Original', image)
    cv2.imshow('Edges (Processed)', closed_edges)
    cv2.imshow('Final Contours', output)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
