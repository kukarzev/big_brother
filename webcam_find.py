if __name__ == '__main__':

    import argparse
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--face', required=True,
                            help='classifier desciption file')
    arg_parser.add_argument('--eyes', help='classifyer for eyes')
    arg_parser.add_argument('device', default=0, type=int,
                            help='which video device to use')
    args = arg_parser.parse_args()

    import cv2

    faceCascade = cv2.CascadeClassifier(args.face)
    webcam = cv2.VideoCapture(args.device)

    if args.eyes:
        eyeCascade = cv2.CascadeClassifier(args.eyes)
    else:
        eyeCascade = None

    go=True
    while go:

        ret, frame = webcam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=5,
            minSize=(frame.shape[1]//10,frame.shape[0]//10))


        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

            if eyeCascade is not None:
                roi_gray = gray[y:y+h, x:x+h]
                eyes = eyeCascade.detectMultiScale(
                    roi_gray, scaleFactor=1.3, minNeighbors=5,
                    minSize=(w//5,h//5))
                for (ex,ey,ew,eh) in eyes:
                    cv2.rectangle(frame, (x+ex,y+ey),(x+ex+ew,y+ey+eh),
                                  (255,0,0), 2)


        cv2.imshow('webcam', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # go=False

    webcam.release()
    cv2.destroyAllWindows()

