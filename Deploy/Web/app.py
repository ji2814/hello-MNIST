import os
import re
from io import BytesIO   

import base64
from flask import Flask, render_template, request, jsonify
import numpy as np
from PIL import Image
import onnxruntime as ort  


app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")


def softmax(x):  
    # 应用softmax函数  
    exps = np.exp(x - np.max(x))  # 减去max(outputs)是为了防止数值溢出  
    probs = exps / np.sum(exps) 
    # 获取预测标签（概率最大的索引）  
    return probs 

def predict(image):
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

    return predicted

@app.route('/predict/', methods=['GET', 'POST'])
def predicted():
    image = parse_image(request.get_data())

    image = image.resize((28, 28)).convert('L')
    
    predicted = predict(image)

    return str(predicted)

# 处理前端传来的image数据
def parse_image(image):
    image_str = re.search(b'base64,(.*)', image).group(1) 
    image_bytes = base64.decodebytes(image_str)  
    image_io = BytesIO(image_bytes)
    image = Image.open(image_io)

    return image


if __name__ == '__main__':
    app.run(debug=True)