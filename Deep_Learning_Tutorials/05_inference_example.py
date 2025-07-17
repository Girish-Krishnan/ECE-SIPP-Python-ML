import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    test_ds = datasets.MNIST(root='data', train=False, download=True, transform=transform)
    loader = DataLoader(test_ds, batch_size=1, shuffle=True)
    image, label = next(iter(loader))
    image = image.to(device)

    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load('cnn_mnist.pt', map_location=device))
    model.eval()
    with torch.no_grad():
        out = model(image)
        pred = out.argmax(dim=1).item()

    print(f'True label: {label.item()} - Predicted: {pred}')


if __name__ == '__main__':
    main()
