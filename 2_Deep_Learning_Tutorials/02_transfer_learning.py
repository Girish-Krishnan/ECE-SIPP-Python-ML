import torch
from torch import nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_ds = datasets.CIFAR10(root='data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(1):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1} done')

    torch.save(model.state_dict(), 'resnet18_cifar10.pt')
    print('Finetuning complete. Model saved as resnet18_cifar10.pt')


if __name__ == "__main__":
    main()
