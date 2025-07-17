import cv2
import torch
from segment_anything import sam_model_registry, SamPredictor


def main():
    sam = sam_model_registry['vit_b'](checkpoint='sam_vit_b.pth')
    predictor = SamPredictor(sam)

    image = cv2.imread('sample.jpg')
    if image is None:
        print('Provide an image named sample.jpg in this folder')
        return

    predictor.set_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Simple rectangular box in the center of the image
    h, w = image.shape[:2]
    input_box = [w//4, h//4, 3*w//4, 3*h//4]
    masks, _, _ = predictor.predict(box=input_box, multimask_output=False)

    mask = masks[0]
    image[mask] = [0, 255, 0]

    cv2.imshow('SAM Segmentation', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
