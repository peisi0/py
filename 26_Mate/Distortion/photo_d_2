import cv2
import numpy as np

DIM = (1920, 1080)
DEFAULT_K = [466.0, 276.0, 960.0, 540.0]
DEFAULT_D = [-0.04, 0.76, -0.61, 0.19]

def nothing(x):
    pass

def main():
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DIM[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DIM[1])

    window_name = "Fisheye Tuner"
    cv2.namedWindow(window_name)

    cv2.createTrackbar("FX", window_name, int(DEFAULT_K[0]), 2000, nothing)
    cv2.createTrackbar("FY", window_name, int(DEFAULT_K[1]), 2000, nothing)
    cv2.createTrackbar("CX", window_name, int(DEFAULT_K[2]), DIM[0], nothing)
    cv2.createTrackbar("CY", window_name, int(DEFAULT_K[3]), DIM[1], nothing)

    cv2.createTrackbar("K1", window_name, int(DEFAULT_D[0]*100 + 1000), 2000, nothing)
    cv2.createTrackbar("K2", window_name, int(DEFAULT_D[1]*100 + 1000), 2000, nothing)
    cv2.createTrackbar("K3", window_name, int(DEFAULT_D[2]*100 + 1000), 2000, nothing)
    cv2.createTrackbar("K4", window_name, int(DEFAULT_D[3]*100 + 1000), 2000, nothing)
    
    cv2.createTrackbar("Zoom", window_name, 100, 200, nothing)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fx = cv2.getTrackbarPos("FX", window_name)
        fy = cv2.getTrackbarPos("FY", window_name)
        cx = cv2.getTrackbarPos("CX", window_name)
        cy = cv2.getTrackbarPos("CY", window_name)
        
        k1 = (cv2.getTrackbarPos("K1", window_name) - 1000) / 100.0
        k2 = (cv2.getTrackbarPos("K2", window_name) - 1000) / 100.0
        k3 = (cv2.getTrackbarPos("K3", window_name) - 1000) / 100.0
        k4 = (cv2.getTrackbarPos("K4", window_name) - 1000) / 100.0
        
        zoom = max(cv2.getTrackbarPos("Zoom", window_name), 1) / 100.0

        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        D = np.array([[k1], [k2], [k3], [k4]], dtype=np.float64)
        
        new_K = K.copy()
        new_K[0, 0] = fx * zoom
        new_K[1, 1] = fy * zoom
        
        try:
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, DIM, cv2.CV_16SC2)
            undistorted = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            display = cv2.resize(undistorted, (960, 540))
            cv2.imshow(window_name, display)
        except cv2.error:
            cv2.imshow(window_name, cv2.resize(frame, (960, 540)))

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            print(f"DIM = {DIM}")
            print(f"K = np.array([[{fx}, 0.0, {cx}], [0.0, {fy}, {cy}], [0.0, 0.0, 1.0]])")
            print(f"D = np.array([[{k1}], [{k2}], [{k3}], [{k4}]])")
            print(f"Zoom = {zoom}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
