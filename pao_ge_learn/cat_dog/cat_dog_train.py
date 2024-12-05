import os

from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms

from pao_ge_learn.googlenet.model import GoogLeNet
from pao_ge_learn.train import train_model_process, plot


def train_valid_data_process(input_size=28, batch_size=128, train_data_rate=0.8):
    data_path = "../data/cat_dog_processed/train"
    compose = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                  transforms.Normalize([0.48613475, 0.45297343, 0.41533014],
                                                       [0.26290178, 0.25554812, 0.25838447])])

    train_data = ImageFolder(data_path, compose)
    print(train_data.class_to_idx)

    num_workers = min(4, os.cpu_count() // 2)

    train_data, valid_data = random_split(train_data, [round(train_data_rate * len(train_data)),
                                                       round((1 - train_data_rate) * len(train_data))])
    train_data_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_data_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return train_data_loader, valid_data_loader


if __name__ == '__main__':


    model = GoogLeNet()
    train_data_loader, valid_data_loader = train_valid_data_process(input_size=224)
    train_process = train_model_process(model, train_data_loader, valid_data_loader, 'best_model.pth',
                                        num_epochs=20, lr=0.001)
    plot(train_process)
