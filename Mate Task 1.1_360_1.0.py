import cv2
import glob


images = []
image_files = glob.glob(r"C:\Users\student\Desktop\folder\*.jpg")
for img_file in image_files:
    img = cv2.imread(img_file)
    images.append(img)

stitcher = cv2.Stitcher_create()


status, panorama = stitcher.stitch(images)

if status == cv2.Stitcher_OK:
    cv2.imwrite("panorama.jpg", panorama)
    print("create sucessfully！")
    a = cv2.imread("panorama.jpg")
    cv2.namedWindow("show", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("show", 800, 600)
    cv2.imshow("show", a)
    if cv2.waitKey(0) == 27:
        cv2.destroyAllWindows();
else:
    print("Failed Code：", status)
