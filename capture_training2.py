# python capture_training2.py --name Gena --training faces/ --N 20 0
#
# this is a point-and-shoot version, takes N pics quickly
#
# python capture_training2.py --N 20 --training faces_pearson/ --name Gena_Kukartsev 1


if __name__=='__main__':
    import argparse
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--name', help='name of person', required=True)
    arg_parser.add_argument('--face',
                            default='haarcascade_frontalface_default.xml',
                            help='classifier desciption file')
    arg_parser.add_argument('--training', help='images for training',
                            required=True)
    arg_parser.add_argument('--N', default=10, type=int, 
                            help='number of images to capture before complete')
    arg_parser.add_argument('device', default=0, type=int,
                            help='which video device to use')
    args = arg_parser.parse_args()

    import cv2
    import time

    faceCascade = cv2.CascadeClassifier(args.face)
    webcam = cv2.VideoCapture(args.device)

    images = []
    go=True
    while go and len(images)<args.N:
        if not webcam.isOpened():
            webcam.open(args.device)

        ret, frame = webcam.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=5,
            minSize=(frame.shape[1]//10, frame.shape[0]//10))

        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

        cv2.imshow('snap', frame)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            go=False
        if len(faces) == 1:
            webcam.release()
            _face = gray[y:y+h, x:x+h]
            _face = cv2.resize(_face, (70,70))
            images.append(_face)
            print('Training image #{} taken'.format(len(images)))
            time.sleep(0.1)

    cv2.destroyAllWindows()

    import os
    tdir = os.path.join(args.training, args.name)
    os.makedirs(tdir, exist_ok=True)
    for i,_face in enumerate(images):
        cv2.imwrite(os.path.join(tdir,'{}.pgm'.format(i)), _face)
