import numpy as np
import cv2
import sys

DIM=(1920, 1080)
K=np.array([[604.7284623795406, 0.0, 953.2403847222108], [0.0, 606.5179385902139, 514.849542398385], [0.0, 0.0, 1.0]])
D=np.array([[-0.010990914174634103], [-0.1284125803723637], [1.6484526914512445], [-2.531658521296613]])
def undistort(img_path):
    img = cv2.imread(img_path)
    h,w = img.shape[:2]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    cv2.imshow("undistorted", undistorted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
if __name__ == '__main__':
    for p in sys.argv[1:]:
        undistort(p)
