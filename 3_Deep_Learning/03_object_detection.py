import torch
from torchvision import transforms
from torchvision.io import read_image
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)
    model.to(device)
    model.eval()

    transform = weights.transforms()
    img = read_image('sample.jpg')  # provide your own image path
    img = transform(img).to(device)

    with torch.no_grad():
        prediction = model([img])[0]

    img = img.cpu().permute(1, 2, 0)
    fig, ax = plt.subplots(1)
    ax.imshow(img)

    for box, score in zip(prediction['boxes'], prediction['scores']):
        if score > 0.8:
            x1, y1, x2, y2 = box.cpu()
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

    plt.show()


if __name__ == "__main__":
    main()
