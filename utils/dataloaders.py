from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

def get_transforms(image_size=224):
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def get_dataloaders(data_dir='data', image_size=224, batch_size=32, num_workers=0):
    train_transform, val_transform = get_transforms(image_size)

    train_dataset = ImageFolder(root=f'{data_dir}/train_small', transform=train_transform)
    val_dataset = ImageFolder(root=f'{data_dir}/val', transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, train_dataset.classes
