import numpy as np
import cv2
import sys
import os

# ================= 你的新参数 =================
DIM = (1920, 1080)
K = np.array([[399.78641805777386, 0.0, 815.0117204817026], 
              [0.0, 309.829035534614, 467.347360718561], 
              [0.0, 0.0, 1.0]])
D = np.array([[0.13337497900058767], [0.455174479504189], 
              [-0.8086347685428306], [0.19747985262163673]])
# ============================================

def test_undistort_manual(img_path):
    if not os.path.exists(img_path): return
    img = cv2.imread(img_path)
    if img is None: return

    dim1 = img.shape[:2][::-1]
    
    # 1. 准备原始 K
    scaled_K = K * dim1[0] / DIM[0]
    scaled_K[2][2] = 1.0

    # 2. 【核心修改】手动构建 new_K，完全绕过自动估算
    # 我们直接用原始 K 的一部分作为新 K，这样绝对不会“爆炸”
    new_K = scaled_K.copy()
    
    # 手动缩放系数：0.4 表示缩小视场，让你能看到更多边缘（类似鱼眼效果）
    # 如果画面太小，可以改成 0.6 或 0.8
    scale_factor = 0.8
    
    new_K[0, 0] *= scale_factor # fx
    new_K[1, 1] *= scale_factor # fy
    
    # 强制让中心点保持在画面正中心
    new_K[0, 2] = dim1[0] / 2 
    new_K[1, 2] = dim1[1] / 2 

    print(f"手动设置缩放: {scale_factor}")

    # 3. 生成映射表
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        scaled_K, D, np.eye(3), new_K, dim1, cv2.CV_16SC2
    )
    
    # 4. 重映射
    undistorted_img = cv2.remap(
        img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )
    
    # 显示
    h, w = undistorted_img.shape[:2]
    show_img = cv2.resize(undistorted_img, (int(w/2), int(h/2)))
    cv2.imshow("Manual Zoom Result", show_img)
    cv2.waitKey(0)

if __name__ == '__main__':
    import glob
    files = glob.glob("*.jpg")
    if len(files) > 0:
        test_undistort_manual(files[0])
    else:
        print("请放一张 jpg 图片")
    cv2.destroyAllWindows()