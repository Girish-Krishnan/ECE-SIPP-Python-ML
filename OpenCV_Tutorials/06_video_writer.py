import cv2


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('Cannot open camera')
        return

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0,
                          (int(cap.get(3)), int(cap.get(4))))

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame. Exiting ...")
            break
        out.write(frame)
        cv2.imshow('Recording (press q to stop)', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print('Video saved as output.avi')


if __name__ == '__main__':
    main()
