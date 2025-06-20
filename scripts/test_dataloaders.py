from utils.dataloaders import get_dataloaders

if __name__ == "__main__":
    train_loader, val_loader, class_names = get_dataloaders()

    print("Classes:", class_names)

    for images, labels in train_loader:
        print("Image batch shape:", images.shape)
        print("Label batch shape:", labels.shape)
        break
