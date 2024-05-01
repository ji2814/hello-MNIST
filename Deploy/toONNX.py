import os, sys
import torch 

# 加载模型
model_dir = os.getcwd() + os.sep + 'Recognize' + os.sep + 'models'
sys.path.append(model_dir)
from LeNet5 import LeNet5
model = LeNet5()

# 加载pth文件
pth_dir = os.path.join(os.getcwd(), "Recognize", "save", "LeNet5.pth")
model.load_state_dict(torch.load(pth_dir)) 

# 准备一个虚拟输入  
dummy_input = torch.randn(1, 1, 28, 28)

# 设置模型为评估模式  
model.eval()  

# 导出模型为ONNX格式  
# onnx文件名
onnx_dir = os.path.join(os.getcwd(), "Deploy", "save", "{}.onnx".format(model._get_name()))
torch.onnx.export(model,               # 模型  
                  dummy_input,          # 模型输入 (或是一个tuple，包含多个输入)  
                  onnx_dir,             # 输出的ONNX文件 
                  export_params=True,   # 存储训练好的参数权重在内  
                  opset_version=11,     # 版本，你可以根据需要更改  
                  do_constant_folding=True,  # 是否执行常量折叠优化  
                  input_names = ['image'],   # 输入张量的名字  
                  output_names = ['label']   # 输出张量的名字
                )