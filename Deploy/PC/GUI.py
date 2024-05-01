import tkinter as tk
import numpy as np
from PIL import Image, ImageDraw
import torch
import train
from tkinter import font
 
# 加载训练好的模型
model = train.Net()
model.load_state_dict(torch.load('best_model.pth', map_location=torch.device('cpu')))
model.eval()

# 创建画布和绘图工具
canvas_width = 280
canvas_height = 280
 
 
def clear_canvas():
    canvas.delete('all')  # 删除画布上的所有元素
    img_draw.rectangle([(20, 0), (300, 280)], fill='black')  # 在图像上绘制一个黑色的矩形覆盖原有内容
    output_label.config(text='')
 
 
def draw(event):
    x, y = event.x, event.y
    canvas.create_text(x, y, text='●', font='Helvetica 20', fill='white')  # 将椭圆填充颜色设为白色
    img_draw.ellipse([(x - 5, y - 5), (x + 5, y + 5)], fill='white')  # 将椭圆描边颜色设为白色
 
 
def recognize_digit():
    # 从画布中获取图像，调整尺寸为模型所需的大小
    image = img.crop((0, 0, 300, 300)).resize((28, 28)).convert('L')
    # 数据预处理
    input_data = np.array(image).reshape((1, 1, 28, 28)).astype('float32')
    input_data = input_data / 255.0  # 将像素值缩放到0到1之间
 
    input_tensor = torch.from_numpy(input_data)
    # 使用模型进行预测
    with torch.no_grad():
        output = model(input_tensor)
        prediction = output.argmax(dim=1).item()
    result_font = font.Font(size=15)
    # 显示预测结果
    output_label.config(text=f'\n识别结果: {prediction}', font=result_font)
 
 
root = tk.Tk()
root.title("手写数字识别")
 
# 计算窗口位置
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
window_width = 320
window_height = 350
x_position = int((screen_width - window_width) / 2)
y_position = int((screen_height - window_height) / 2)
root.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")
 
canvas = tk.Canvas(root, width=canvas_width, height=canvas_height, bg='black')
 
 
# 固定窗口大小
root.resizable(False, False)
root.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")
 
canvas = tk.Canvas(root, width=canvas_width, height=canvas_height, bg='black')
canvas.pack()
 
img = Image.new('RGB', (canvas_width, canvas_height), 'black')
img_draw = ImageDraw.Draw(img)
 
canvas.bind('<B1-Motion>', draw)
 
recognize_button = tk.Button(root, text="识别", command=recognize_digit, width=10, height=2)
recognize_button.pack(side=tk.LEFT, padx=10, pady=10)
 
clear_button = tk.Button(root, text="清除", command=clear_canvas, width=10, height=2)
clear_button.pack(side=tk.RIGHT, padx=10, pady=10)
 
output_label = tk.Label(root, text='', font=('Arial', 20))
output_label.pack()
 
root.mainloop()