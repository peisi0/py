import tkinter as tk
from PIL import Image, ImageTk

class PanoramicViewer:
    def __init__(self, master, image_path):
        self.master = master
        self.canvas = tk.Canvas(master, width=800, height=400)
        self.canvas.pack()

        # 加载全景图像
        self.image = Image.open(image_path)
        self.tk_image = ImageTk.PhotoImage(self.image)

        # 创建图像对象
        self.image_obj = self.canvas.create_image(0, 0, anchor='nw', image=self.tk_image)

        # 绑定事件
        self.canvas.bind('<ButtonPress-1>', self.on_press)
        self.canvas.bind('<B1-Motion>', self.on_drag)

        self.canvas.config(scrollregion=self.canvas.bbox(self.image_obj))  # 设置滚动区域

    def on_press(self, event):
        self.x = event.x
        self.y = event.y

    def on_drag(self, event):
        dx = event.x - self.x
        dy = event.y - self.y
        self.canvas.move(self.image_obj, dx, dy)
        self.x = event.x
        self.y = event.y

# 创建主窗口
root = tk.Tk()
root.title("Image showing")
#root.iconbitmap(r"C:\Users\student\Downloads\fff.png")
app = PanoramicViewer(root, r"C:\Users\student\Desktop\panorama.jpg")

# 运行主循环
root.mainloop()
