# python webcam_find.py --training orl_faces 0



if __name__ == '__main__':

    import argparse
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--face',
                            default='haarcascade_frontalface_default.xml',
                            help='classifier desciption file')
    arg_parser.add_argument('--eyes', help='classifier for eyes')
    arg_parser.add_argument('device', default=0, type=int,
                            help='which video device to use')
    arg_parser.add_argument('--training', help='images for training',
                            required=True)
    # for motion detection
    arg_parser.add_argument("-v", "--video", help="path to the video file")
    arg_parser.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")

    args = arg_parser.parse_args()
    
       
    # initialize the first frame in the video stream
    firstFrame = None

    import cv2
    import recognition as r
    import numpy as np
    import time, datetime
    import greetings
    import threading

    # text-to-speech init
    g = greetings.Greetings()

    # read test data
    [XX,yy], names = r.read_images(args.training, sz=(70,70))
    yy = np.asarray(yy, dtype=np.int)
    model = cv2.face.createFisherFaceRecognizer()
    print('About to train the recognizer, the available labels:',names)
    model.train(np.asarray(XX), np.asarray(yy))
    print('Recognize trained...')

    faceCascade = cv2.CascadeClassifier(args.face)
    #webcam = cv2.VideoCapture(args.device)

    if args.eyes:
        eyeCascade = cv2.CascadeClassifier(args.eyes)
    else:
        eyeCascade = None

    # if the video argument is not None, then we are reading from a video file
    if args.video:
        webcam = cv2.VideoCapture(args.video)
    # otherwise, we are reading from webcam
    else:
        webcam = cv2.VideoCapture(args.device)
        time.sleep(0.25)

        
    # starting epoch time
    _began = time.time()
    _last_reset = _began
    _reset_reference = False
    _last_prediction_time = _began
    _last_prediction = set()
    _last_unoccupied = _began
    _last_announcement = _began-5.0
    _last_greeted = set()
    go=True

    face_history = {}
    history_size = 30
    while go:

        ret, frame = webcam.read()



        # motion detection piece
        text = 'Unoccupied'
        if not ret:
            break # reached the end of video

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _blurred_gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # check if it is time to reset reference frame
        if (time.time() - _last_reset) > 1.0:
            text = "Unoccupied"
            _reset_reference = True
            _last_reset = time.time()

        if (firstFrame is None) or _reset_reference:
            firstFrame = _blurred_gray
            _reset_reference = False
            continue

        # compute the absolute difference between the current frame and
        # first frame
        frameDelta = cv2.absdiff(firstFrame, _blurred_gray)
        thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
        
        # dilate the thresholded image to fill in holes, then find contours
        # on thresholded image
        thresh = cv2.dilate(thresh, None, iterations=2)
        (_, cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	# loop over the contours
        for c in cnts:
            # if the contour is too small, ignore it
            if cv2.contourArea(c) < args.min_area:
                _last_unoccupied = time.time()
                continue
            
            # compute the bounding box for the contour, draw it on the frame,
            # and update the text
            #(x, y, w, h) = cv2.boundingRect(c)
            #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = "Occupied"

        # draw the text and timestamp on the frame
        cv2.putText(frame, "Room Status: {}".format(text), (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                    (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        

        # consider saying something
        if ((_last_prediction_time < _last_unoccupied) and
            len(_last_prediction-_last_greeted) > 0 and 
            (time.time()-_last_announcement)>5.0):
            _last_announcement = time.time()
            people = ' and '.join(_last_prediction - _last_greeted)
            t = threading.Thread(target=g.Greet, args=('Hello, {}'.format(
                people),))
            t.start()
            _last_greeted = set(_last_prediction)


        if text=='Occupied':
            faces = faceCascade.detectMultiScale(
                gray, scaleFactor=1.3, minNeighbors=5,
                minSize=(frame.shape[1]//10,frame.shape[0]//10))

            if len(faces) > 0:
                _last_prediction.clear()

            for i,(x, y, w, h) in enumerate(faces):
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
                
                # get the face
                _face = gray[y:y+h, x:x+h]

                if eyeCascade is not None:
                    eyes = eyeCascade.detectMultiScale(
                        _face, scaleFactor=1.3, minNeighbors=5,
                        minSize=(w//5,h//5))
                    for (ex,ey,ew,eh) in eyes:
                        cv2.rectangle(frame, (x+ex,y+ey),(x+ex+ew,y+ey+eh),
                                      (255,0,0), 2)


                # resize face
                _width = 70
                _height = 70
                _face_resized = cv2.resize(_face, (_width,_height))

                # predict
                _prediction = model.predict(_face_resized)

                hist = face_history.get(i, np.zeros(history_size,
                                                    dtype=np.int64))
                hist = np.concatenate((hist[1:], np.array([_prediction[0]])))
                face_history[i] = hist
                
                try:
                    counts = np.bincount(hist)
                    maxBin = np.argmax(counts)
                except:
                    print(hist)
                    maxBin = None

                if maxBin and (counts[maxBin] >= history_size*0.5):
                    _label = names[maxBin]
                else:
                    _label = 'Stranger'

                _last_prediction.add(_label)
                _last_prediction_time = time.time()

                # add some text
                pos_x = max(x-10,0)
                pos_y = max(y-10,0)
                #cv2.putText(frame, '{} {}'.format(_label, _prediction),
                cv2.putText(frame, '{}'.format(_label),
                        (pos_x,pos_y), cv2.FONT_HERSHEY_PLAIN,1,(0,255,0),
                        2)


        

        cv2.imshow('webcam', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # go=False

    webcam.release()
    cv2.destroyAllWindows()

