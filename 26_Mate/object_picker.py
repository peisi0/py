import cv2
import numpy as np

def process_image(str):
    # 1. 读取图片
    img = cv2.imread(str)
    if img is None:
        print("找不到图片")
        return

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([80, 40, 50])
    upper_blue = np.array([130, 255, 255])
    mask_ipad = cv2.inRange(hsv, lower_blue, upper_blue)

    lower_orange = np.array([10, 100, 100])
    upper_orange = np.array([25, 255, 255])
    mask_phone = cv2.inRange(hsv, lower_orange, upper_orange)

    combined_mask = cv2.bitwise_or(mask_ipad, mask_phone)

    kernel = np.ones((15, 15), np.uint8) 
    mask_closed = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    mask_clean = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)

    # 6. 寻找轮廓
    contours, hierarchy = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 复制一份原图用来画线
    output = img.copy()

    print(f"检测到的轮廓数量: {len(contours)}")

    for cnt in contours:
        # 7. 面积过滤 (Area Filter)
        # 计算轮廓面积
        area = cv2.contourArea(cnt)

        # 这里的 5000 是个阈值，小于这个面积的（比如误识别的杂点）都忽略
        if area > 5000: 
            # 为了画出好看的框，我们用最小外接矩形
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            
            
            box = np.int32(box) 
            
            cv2.drawContours(output, [box], 0, (0, 0, 255), 3)
            
            center_x, center_y = int(rect[0][0]), int(rect[0][1])
            cv2.putText(output, f"Obj Area: {int(area)}", (center_x - 50, center_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    
    output_small = cv2.resize(output, (0,0), fx=0.4, fy=0.4)
    mask_small = cv2.resize(mask_clean, (0,0), fx=0.4, fy=0.4)

    cv2.imshow('Processed Mask', mask_small) 
    cv2.imshow('Final Result', output_small)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_image("image.jpg")
