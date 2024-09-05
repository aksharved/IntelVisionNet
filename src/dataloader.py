import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def load_data(batch_size, data_dir):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224 for CNN input
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_data = datasets.ImageFolder(root=f"{data_dir}/seg_train/seg_train", transform=transform)
    test_data = datasets.ImageFolder(root=f"{data_dir}/seg_test/seg_test", transform=transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
