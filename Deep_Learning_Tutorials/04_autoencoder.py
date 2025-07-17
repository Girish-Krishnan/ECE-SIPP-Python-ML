import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 64),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 28 * 28),
            nn.Sigmoid(),
            nn.Unflatten(1, (1, 28, 28))
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.ToTensor()

    train_ds = datasets.MNIST(root='data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)

    model = AutoEncoder().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1):
        for images, _ in train_loader:
            images = images.to(device)
            optimizer.zero_grad()
            recon = model(images)
            loss = criterion(recon, images)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1} done')

    torch.save(model.state_dict(), 'autoencoder_mnist.pt')
    print('Model saved as autoencoder_mnist.pt')


if __name__ == '__main__':
    main()
