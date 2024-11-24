import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torchvision.transforms import transforms

train_data = FashionMNIST(root='../data',
                          train=True,
                          transform=transforms.Compose([transforms.Resize(size=224), transforms.ToTensor()]),
                          download=True)

train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True, num_workers=0)

for step, (x, y) in enumerate(train_loader):
    if step > 0:
        break
bx = x.squeeze().numpy()  # 将4维张量移除第1维，转成numpy格式
by = y.numpy()
label = train_data.classes
print(label)

rows = 8
cols = len(by) // rows if len(by) % rows == 0 else (len(by) // rows + 1)
plt.figure(figsize=(rows, cols))
for i in range(len(by)):
    plt.subplot(rows, cols, i + 1)
    plt.imshow(bx[i, ::, ::], cmap="gray")
    plt.title(label[by[i]], size=10)  # 缩小标题字体
    plt.axis('off')
    plt.subplots_adjust(wspace=0.05, hspace=0.6)  # 调整水平和垂直间距
plt.show()
