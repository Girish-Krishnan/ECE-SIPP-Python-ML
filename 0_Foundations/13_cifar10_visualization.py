"""Download and visualize images from the CIFAR-10 dataset."""
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def main():
    # Download CIFAR-10 training data
    transform = transforms.ToTensor()
    cifar = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)

    print("Number of images:", len(cifar))

    loader = DataLoader(cifar, batch_size=4, shuffle=True)
    images, labels = next(iter(loader))

    # Plot a few sample images
    fig, axes = plt.subplots(1, 4, figsize=(8, 2))
    for img, ax in zip(images, axes):
        ax.imshow(img.permute(1, 2, 0))
        ax.axis("off")
    plt.suptitle("CIFAR-10 samples")
    plt.show()


if __name__ == "__main__":
    main()
