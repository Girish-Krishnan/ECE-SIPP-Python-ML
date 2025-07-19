import cv2
import pytesseract


def main():
    image = cv2.imread('ocr_sample.png')
    if image is None:
        print('Provide an image named ocr_sample.png in this folder')
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)

    print('Recognized text:')
    print(text)


if __name__ == '__main__':
    main()
