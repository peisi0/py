import numpy as np
import cv2
import sys

DIM = (1920, 1080)
K=np.array([[604.7284623795406, 0.0, 953.2403847222108], [0.0, 606.5179385902139, 514.849542398385], [0.0, 0.0, 1.0]])
D=np.array([[-0.010990914174634103], [-0.1284125803723637], [1.6484526914512445], [-2.531658521296613]])

def undisort(img_path, balance = 0.0, dim1 = None, dim2 = None, dim3 = None):
    img = cv2.imread(img_path)
    dim1 = img.shape[:2][::-1]
    assert dim1[0]/dim1[1] == DIM[0]/DIM[1], "Error: 1001"
    if not dim2:
        dim2 = dim1
    if not dim3:
        dim3 = dim1
    scaled_K = K * dim1[0] / DIM[0]
    scaled_K[2][2] = 1.0
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D, dim2, np.eye(3), balance = balance)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3), new_K, dim3, cv2.CV_16SC2)
    undisorted_img = cv2.remap(img, map1, map2, interpolation = cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    cv2.imshow("undisorted", undisorted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    for p in sys.argv[1:]:
        undisort(p)
