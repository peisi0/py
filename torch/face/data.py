import cv2
import os
import numpy as np

# --- 配置参数 ---
# 保存数据集的根目录
DATASET_PATH = "my_digits"
# 我们希望采集的图像大小
IMG_SIZE = 28


def create_folder(folder_path):
    """如果文件夹不存在，则创建它"""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"文件夹 '{folder_path}' 已创建")


def main():
    """主函数，用于采集和保存手写数字图像"""
    create_folder(DATASET_PATH)

    while True:
        # 提示用户输入要采集的数字
        digit_to_capture = input("请输入您要采集的数字 (0-9)，或输入 'q' 退出: ")
        if digit_to_capture.lower() == 'q':
            break
        if not digit_to_capture.isdigit() or not 0 <= int(digit_to_capture) <= 9:
            print("无效输入，请输入一个0到9之间的数字。")
            continue

        # 为当前数字创建子文件夹
        digit_folder = os.path.join(DATASET_PATH, digit_to_capture)
        create_folder(digit_folder)

        # 获取该文件夹下已有的样本数量，用于文件命名
        count = len(os.listdir(digit_folder))

        # 打开摄像头
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("错误：无法打开摄像头。")
            return

        print(f"\n准备为数字 '{digit_to_capture}' 采集样本...")
        print("将数字放在摄像头前，按 's' 保存样本。")
        print("按 'q' 结束当前数字的采集。")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            display_frame = frame.copy()

            # 图像预处理，与实时检测时相同
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)[1]
            contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # 从二值图像中提取ROI
                roi = thresh[y:y + h, x:x + w]
                if roi.size > 0:
                    # 调整大小并保存
                    processed_roi = cv2.resize(roi, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('s'):
                        # 保存处理后的ROI图像
                        save_path = os.path.join(digit_folder, f"{count}.png")
                        cv2.imwrite(save_path, processed_roi)
                        print(f"已保存样本: {save_path}")
                        count += 1
                    elif key == ord('q'):
                        break

            cv2.imshow(f"'{digit_to_capture}' detecting... (s=save, q=done)", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
