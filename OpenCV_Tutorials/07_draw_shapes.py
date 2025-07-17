import cv2
import numpy as np


def main():
    canvas = np.zeros((400, 400, 3), dtype=np.uint8)

    cv2.rectangle(canvas, (50, 50), (150, 150), (0, 255, 0), 2)
    cv2.circle(canvas, (300, 100), 50, (255, 0, 0), -1)
    cv2.line(canvas, (50, 300), (350, 300), (0, 0, 255), 3)
    cv2.putText(canvas, 'OpenCV!', (180, 250), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 255, 255), 2)

    cv2.imshow('Shapes', canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
