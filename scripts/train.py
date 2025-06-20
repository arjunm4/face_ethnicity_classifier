import torch
import torch.nn as nn
import torch.optim as optim
from models.ethnicity_model import get_model
from utils.dataloaders import get_dataloaders

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10):
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        print(f"\n🚀 Epoch {epoch + 1}/{num_epochs}")
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            # 🔁 Print batch info every 10 batches
            if i % 10 == 0:
                print(f"  [Batch {i:03d}] Loss: {loss.item():.4f}")

            # Debug: break early if needed
            # if i == 20:
            #     break

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_corrects = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        val_loss /= len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset)

        print(f"\n📊 Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f}")
        print(f"🧪  Val  Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print("✅ Best model saved!")

    print("\n🎉 Training complete")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, class_names = get_dataloaders()
    model = get_model(num_classes=len(class_names)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10)
