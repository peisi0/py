import numpy as np
import cv2
import sys

DIM=(1920, 1080)
K=np.array([[193.34482383427405, 0.0, 1069.4157872430894], [0.0, 188.35827200859535, 616.3583154874141], [0.0, 0.0, 1.0]])
D=np.array([[-0.041826625553906656], [0.1566823574305782], [-0.2606611137597746], [-0.026691365823080487]])
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
