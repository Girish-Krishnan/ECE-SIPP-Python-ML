import cv2


def main():
    back_sub = cv2.createBackgroundSubtractorMOG2()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame. Exiting ...")
            break
        fg_mask = back_sub.apply(frame)
        cv2.imshow('Background Subtraction', fg_mask)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
