import os
import torch
import torchvision

from models.MLP import MLP
from models.LeNet5 import LeNet5
from models.ResNet import ResNet

# 加载MNIST数据集
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=torchvision.transforms.ToTensor())

# 定义数据加载器
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# 定义模型
# net = LeNet5()
# net = MLP()
net = ResNet()

# 加载模型参数
model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'save', net._get_name() + '.pth')
net.load_state_dict(torch.load(model_dir))

# 测试模型
net.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy: {:.2f}%'.format(100 * correct / total))