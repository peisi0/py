import cv2
import glob
import ctypes
import tkinter as tk
from PIL import Image, ImageTk

#class PanoramicViewer:
#    def __init__(self, master, image_path):
#        self.master = master
#        self.canvas = tk.Canvas(master, width=800, height=400)
#        self.canvas.pack()
#
#        
#        self.image = Image.open(image_path)
#        self.tk_image = ImageTk.PhotoImage(self.image)
#
#        
#        self.image_obj = self.canvas.create_image(0, 0, anchor='nw', image=self.tk_image)
#
#        
#        self.canvas.bind('<ButtonPress-1>', self.on_press)
#        self.canvas.bind('<B1-Motion>', self.on_drag)
#
#        self.canvas.config(scrollregion=self.canvas.bbox(self.image_obj))  # 设置滚动区域
#
#    def on_press(self, event):
#        self.x = event.x
#        self.y = event.y
#
#    def on_drag(self, event):
#        dx = event.x - self.x
#        dy = event.y - self.y
#        self.canvas.move(self.image_obj, dx, dy)
#        self.x = event.x
#        self.y = event.y

images = []


stitcher = cv2.Stitcher_create()

cap = cv2.VideoCapture(1)


if not cap.isOpened():
    print("无法打开摄像头")
    exit()
i = 0
while True:
    
    ret, frame = cap.read()
    if not ret:
        print("无法读取帧")
        break
    k = cv2.waitKey(1)
    if k == ord("1"):
        break
    elif k == ord("s"):
        cv2.imwrite(r"C:\Users\student\Desktop\photos\\"+str(i)+".jpg", frame)
        i += 1
    elif k == ord("n"):
        image_files = glob.glob(r"C:\Users\student\Desktop\photos\*.jpg")
        for img_file in image_files:
            img = cv2.imread(img_file)
            images.append(img)

        status, panorama = stitcher.stitch(images)

        if status == cv2.Stitcher_OK:
            cv2.imwrite(r"C:\Users\student\Desktop\photos\photosphere.png", panorama)
            print("create sucessfully！")
            a = cv2.imread(r"C:\Users\student\Desktop\photos\photosphere.png")
            cv2.destroyWindow('Camera')
            cv2.namedWindow("show", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("show", ctypes.windll.user32.GetSystemMetrics(0), 350)
            cv2.imshow("show", a)
            if cv2.waitKey(0) == 27:
                cv2.destroyAllWindows();
            #root = tk.Tk()
            #root.title("Image showing")
            ##root.iconbitmap(r"C:\Users\student\Downloads\fff.png")
            #app = PanoramicViewer(root, r"C:\Users\student\Desktop\photos\photosphere.png")

    cv2.imshow('Camera', frame)

cap.release()
cv2.destroyAllWindows()
