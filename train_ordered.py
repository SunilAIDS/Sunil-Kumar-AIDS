import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import RetinaNet

# -------------------- Hyperparameters --------------------
epochs = 25
batch_size = 16
lr = 1e-4
num_classes = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- Dataset Transforms --------------------
train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# -------------------- Load Dataset --------------------
train_dataset = datasets.ImageFolder('datasets/train', transform=train_transform)
val_dataset = datasets.ImageFolder('datasets/val', transform=val_transform)

# Ensure class order
desired_classes = ['amd','cataract','dr','glaucoma','healthy']
train_dataset.class_to_idx = {cls: idx for idx, cls in enumerate(desired_classes)}
val_dataset.class_to_idx = {cls: idx for idx, cls in enumerate(desired_classes)}

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# -------------------- Model --------------------
model = RetinaNet(num_classes=num_classes).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# -------------------- Training Loop --------------------
for epoch in range(epochs):
    model.train()
    running_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {running_loss/len(train_loader):.4f}")

    # Validation
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            preds = torch.argmax(outputs,1)
            correct += (preds==labels).sum().item()
            total += labels.size(0)
    print(f"Validation Loss: {val_loss/len(val_loader):.4f} | Accuracy: {correct/total*100:.2f}%")

# -------------------- Save Model --------------------
torch.save(model.state_dict(), "retina_model.pth")
print("Model saved as retina_model.pth")
