import os, sys

import tkinter as tk
import numpy as np
from PIL import Image, ImageDraw
from tkinter import font
import onnxruntime as ort  

# 创建画布和绘图工具
canvas_width = 280
canvas_height = 280

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

# 固定窗口大小
root.resizable(False, False)
root.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")

# 创建画布
canvas = tk.Canvas(root, width=canvas_width, height=canvas_height, bg='black')
canvas.pack()
# 创建绘画背景
img = Image.new('RGB', (canvas_width, canvas_height), 'black')
img_draw = ImageDraw.Draw(img)


def softmax(x):  
    # 应用softmax函数  
    exps = np.exp(x - np.max(x))  # 减去max(outputs)是为了防止数值溢出  
    probs = exps / np.sum(exps) 
    # 获取预测标签（概率最大的索引）  
    return probs 

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

    '''数据预处理'''
    input_data = np.array(image).reshape((1, 1, 28, 28)).astype('float32')
    input_data = input_data / 255.0  # 将像素值缩放到0到1之间

    # 标准化
    mean = np.array([0.5])  # 均值  
    std = np.array([0.5])   # 标准差  
    
    # 扩展mean和std的维度以匹配图片的通道数  
    mean = mean.reshape(1, 1, 1, 1)  
    std = std.reshape(1, 1, 1, 1)    
    
    # 使用numpy的广播机制进行标准化  
    input_data = (input_data - mean) / std 
    input_data = input_data.astype('float32')

    '''使用ONNX模型预测'''
    # 设置ONNX模型文件的路径  
    onnx_model_path = os.path.join(os.getcwd(), "Deploy", "save", "LeNet5.onnx")  
    
    # 创建ONNX Runtime会话  
    ort_session = ort.InferenceSession(onnx_model_path)  
    
    # 获取输入和输出的名称  
    input_name = ort_session.get_inputs()[0].name  
    output_name = ort_session.get_outputs()[0].name 

    # 推理获取输出
    outputs = ort_session.run([output_name], {input_name: input_data})    
    # 假设模型只有一个输出，并且我们想要获取这个输出  
    output_data = outputs[0]  

    probs = softmax(output_data)
    predicted = np.argmax(probs)  

    # 显示预测结果
    output_label.config(text=f'\n识别结果: {predicted}', font=font.Font(size=15))

# 绑定鼠标左键
canvas.bind('<B1-Motion>', draw)

# 识别按钮
recognize_button = tk.Button(root, text="识别", command=recognize_digit, width=10, height=2)
recognize_button.pack(side=tk.LEFT, padx=10, pady=10)
 
# 清除按钮
clear_button = tk.Button(root, text="清除", command=clear_canvas, width=10, height=2)
clear_button.pack(side=tk.RIGHT, padx=10, pady=10)

# 预测标签
output_label = tk.Label(root, text='', font=('Arial', 20))
output_label.pack()
 
# 主循环启动
root.mainloop()