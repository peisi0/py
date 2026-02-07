import cv2
import os
import time

# ================= 配置区域 =================
# 保存图片的文件夹名称
SAVE_FOLDER = "calibration_images"
# 摄像头分辨率 (必须与你后续标定的 DIM 一致)
WIDTH = 1920
HEIGHT = 1080
# ===========================================

def create_folder():
    if not os.path.exists(SAVE_FOLDER):
        os.makedirs(SAVE_FOLDER)
        print(f"文件夹 '{SAVE_FOLDER}' 已创建")

def draw_grid(img, w, h):
    """画九宫格辅助线，帮助你构图"""
    color = (0, 255, 0) # 绿色
    thickness = 1
    
    # 垂直线
    cv2.line(img, (int(w/3), 0), (int(w/3), h), color, thickness)
    cv2.line(img, (int(w*2/3), 0), (int(w*2/3), h), color, thickness)
    # 水平线
    cv2.line(img, (0, int(h/3)), (w, int(h/3)), color, thickness)
    cv2.line(img, (0, int(h*2/3)), (w, int(h*2/3)), color, thickness)
    
    # 中心十字
    cv2.line(img, (int(w/2)-20, int(h/2)), (int(w/2)+20, int(h/2)), (0, 0, 255), 2)
    cv2.line(img, (int(w/2), int(h/2)-20), (int(w/2), int(h/2)+20), (0, 0, 255), 2)

def main():
    create_folder()
    
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW) 
    
    # 强制设置分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    
    # 检查实际分辨率
    actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"摄像头已启动 | 分辨率: {int(actual_w)}x{int(actual_h)}")
    
    if actual_w != WIDTH or actual_h != HEIGHT:
        print(f"警告：摄像头不支持 {WIDTH}x{HEIGHT}，当前为 {int(actual_w)}x{int(actual_h)}")

    count = 0
    last_save_time = 0
    show_msg_duration = 1.0 # 保存提示显示 1 秒

    print("\n操作指南:")
    print("  [S] 键: 拍照保存")
    print("  [Q] 键: 退出程序")
    print("-----------------------------------")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法获取画面，请检查摄像头连接")
            break

        # 复制一份用于显示（避免把辅助线保存进图片里）
        display_frame = frame.copy()
        
        # 画辅助线
        draw_grid(display_frame, int(actual_w), int(actual_h))
        
        # 显示操作提示
        cv2.putText(display_frame, f"Count: {count}  |  Press 'S' to Save, 'Q' to Quit", 
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # 如果刚保存过，显示“已保存”提示
        if time.time() - last_save_time < show_msg_duration:
            cv2.putText(display_frame, "SAVED!", (int(actual_w/2)-100, int(actual_h/2)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)

        cv2.imshow("Calibration Capture Tool", display_frame)

        key = cv2.waitKey(1) & 0xFF

        # 按 's' 保存
        if key == ord('s'):
            filename = os.path.join(SAVE_FOLDER, f"calib_{count:03d}.jpg")
            # 注意：保存的是原始 frame，没有辅助线
            cv2.imwrite(filename, frame)
            print(f"已保存: {filename}")
            count += 1
            last_save_time = time.time() # 记录保存时间用于显示提示
        
        # 按 'q' 退出
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()