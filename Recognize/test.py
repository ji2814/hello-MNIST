import os
import torch
import torchvision
import matplotlib.pyplot as plt

from models._import import LeNet5, MLP, ResNet, GRU, ViT

# 加载MNIST数据集
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=torchvision.transforms.ToTensor())

# 定义数据加载器
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# 定义模型
net = LeNet5()

# 加载模型参数
model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'save', net._get_name() + '.pth')
net.load_state_dict(torch.load(model_dir))

# 计算模型精度
net.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for image, label in test_loader:
        outputs = net(image)
        _, predicted = torch.max(outputs.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()

    accuracy = 100 * correct / total
    print('Test Accuracy: {:.2f}%'.format(accuracy))

# 加载测试集
examples = enumerate(test_loader)
_, (example_images, example_targets) = next(examples)

# 显示图片
fig = plt.figure()
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.tight_layout()
    plt.imshow(example_images[i][0], cmap='gray', interpolation='none')

    example_outputs = net(example_images)
    _, example_predicts = torch.max(example_outputs.data, dim=1)
    
    plt.title("Ground Truth: {} \n Predict: {}".format(example_targets[i], example_predicts[i]))

    '''移除刻度'''
    plt.xticks([])
    plt.yticks([])

# 显示精确度
accuracy_text = "Accuracy: {:.2f}%".format(accuracy)
plt.figtext(0.5, 0.01, accuracy_text, ha='center', fontsize=12)

plt.show()