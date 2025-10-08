import cv2
import numpy as np

# 读取视频
cap = cv2.VideoCapture('video.mp4')

# 读取第一帧
ret, frame = cap.read()

# 设置初始窗口 (x, y, width, height)
x, y, w, h = 300, 200, 100, 50
track_window = (x, y, w, h)

# 设置 ROI (Region of Interest)
roi = frame[y:y+h, x:x+w]

# 转换为 HSV 颜色空间
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

# 创建掩膜并计算直方图
mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

# 设置终止条件
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 转换为 HSV 颜色空间
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 计算反向投影
    dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

    # 应用 MeanShift 算法
    ret, track_window = cv2.meanShift(dst, track_window, term_crit)

    # 绘制跟踪结果
    x, y, w, h = track_window
    img2 = cv2.rectangle(frame, (x, y), (x+w, y+h), 255, 2)
    cv2.imshow('MeanShift Tracking', img2)

    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()