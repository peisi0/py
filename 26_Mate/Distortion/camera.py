import cv2
import numpy as np
import glob
import sys

# 1. 确认棋盘格参数
CHECKERBOARD = (9, 6) 

subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
# calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW
calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW + cv2.fisheye.CALIB_USE_INTRINSIC_GUESS


objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

objpoints = [] 
imgpoints = [] 

# 2. 检查图片加载
images = glob.glob('*.jpg')
print(f"检测到图片数量: {len(images)}")

if len(images) == 0:
    print("错误：未找到图片！请检查文件后缀（.png/.jpg）或路径。")
    sys.exit()

_img_shape = None
valid_images = 0

for fname in images:
    img = cv2.imread(fname)
    if img is None:
        print(f"无法读取图片: {fname}")
        continue
        
    if _img_shape is None:
        _img_shape = img.shape[:2]
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 寻找角点
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, 
        cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
    
    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (3,3), (-1,-1), subpix_criteria)
        imgpoints.append(corners2)
        valid_images += 1
        print(f"[成功] {fname} 检测到角点")
    else:
        print(f"[失败] {fname} 未检测到角点 - 请检查 CHECKERBOARD 尺寸")

# 3. 关键检查：如果没有有效图片，不要运行 calibrate
if valid_images == 0:
    print("\n严重错误：没有一张图片成功检测到角点！")
    print("原因可能是：")
    print(f"1. CHECKERBOARD={CHECKERBOARD} 设置错误（请数内部十字交叉点，不是格子数）")
    print("2. 图片太模糊或畸变太严重")
    sys.exit()

print(f"\n开始标定，使用有效图片: {valid_images} 张...")

N_OK = len(objpoints)
K = np.zeros((3, 3))
D = np.zeros((4, 1))
# rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
# tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]

h, w = _img_shape
K[0, 0] = w / 2.0  # 假设焦距 fx 约为宽度的一半 (粗略估计)
K[1, 1] = w / 2.0  # 假设焦距 fy 约为宽度的一半
K[0, 2] = w / 2.0  # 主点 cx 在图像中心
K[1, 2] = h / 2.0  # 主点 cy 在图像中心
K[2, 2] = 1.0

rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]


rms, _, _, _, _ = cv2.fisheye.calibrate(
    objpoints,
    imgpoints,
    _img_shape[::-1],
    K,
    D,
    rvecs,
    tvecs,
    calibration_flags,
    (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
)

print("标定成功！")
print("DIM = " + str(_img_shape[::-1]))
print("K =", K.tolist())
print("D =", D.tolist())
