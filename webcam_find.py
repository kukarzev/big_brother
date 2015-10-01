# python webcam_find.py --face ../opencv/data/haarcascades/haarcascade_frontalface_default.xml --training orl_faces 0



if __name__ == '__main__':

    import argparse
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--face', required=True,
                            help='classifier desciption file')
    arg_parser.add_argument('--eyes', help='classifier for eyes')
    arg_parser.add_argument('device', default=0, type=int,
                            help='which video device to use')
    arg_parser.add_argument('--training', help='images for training')
    args = arg_parser.parse_args()

    import cv2
    import recognition as r
    import numpy as np

    # read test data
    [XX,yy] = r.read_images(args.training, sz=(70,70))
    yy = np.asarray(yy, dtype=np.int32)
    model = cv2.face.createFisherFaceRecognizer()
    print('About to train the recognizer, the available labels:',np.asarray(yy))
    model.train(np.asarray(XX), np.asarray(yy))
    print('Recognize trained...')

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


        for i,(x, y, w, h) in enumerate(faces):
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            
            if eyeCascade is not None:
                roi_gray = gray[y:y+h, x:x+h]
                eyes = eyeCascade.detectMultiScale(
                    roi_gray, scaleFactor=1.3, minNeighbors=5,
                    minSize=(w//5,h//5))
                for (ex,ey,ew,eh) in eyes:
                    cv2.rectangle(frame, (x+ex,y+ey),(x+ex+ew,y+ey+eh),
                                  (255,0,0), 2)

            # get the face
            _face = gray[y:y+h, x:x+h]

            # resize face
            _width = 70
            _height = 70
            _face_resized = cv2.resize(_face, (_width,_height))

            # predict
            _prediction = model.predict(_face_resized)

            # add some text
            pos_x = max(x-10,0)
            pos_y = max(y-10,0)
            cv2.putText(frame,'Prediction: {}'.format(_prediction), (pos_x,pos_y), cv2.FONT_HERSHEY_PLAIN,1,(0,255,0), 2)

        cv2.imshow('webcam', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # go=False

    webcam.release()
    cv2.destroyAllWindows()

