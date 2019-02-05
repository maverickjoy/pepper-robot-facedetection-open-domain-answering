import os
import cv2
import math
import time
import pickle
import datetime
import numpy as np
from sklearn import neighbors
from PIL import Image, ImageDraw
import face_recognition.api as face_recognition

####################################
# DEVELOPER        : VIKRAM SINGH  #
# TECHNOLOGY STACK : PYTHON        #
####################################


# =============================== CONFIG =======================================

RECOGNISED_NAMES         = "recorded_data/names.txt" # saved names list of people recognised
RECOGNISED_ENCODINGS     = "recorded_data/encodings.txt" # saved encoding list of people recognised
MODEL_PATH               = "recorded_data/models/model.txt" # tranied model to save path
DIST_THRESHOLD           = 0.45 # Threshold value iversely proportional to accuracy
DETECTED_IMAGE_SAVE_PATH = "recorded_data/images/detections/"

# ==============================================================================


class FaceDetector():

    def __init__(self):

        self.frameNumber         = 1
        self.start_time          = time.time()
        self.knownNames          = []
        self.knownFaceEncodings  = []

    '''
    Retrain model for any new entries of faces
    '''
    def scanKnownFaces(self):
        with open(RECOGNISED_NAMES) as f:
            known_names = f.read().splitlines()

        known_face_encodings=np.loadtxt(RECOGNISED_ENCODINGS, dtype='float')

        n_neighbors = int(round(math.sqrt(len(known_face_encodings))))
        knn_clf = neighbors.KNeighborsClassifier(n_neighbors, algorithm='ball_tree', weights='distance')
        # print known_face_encodings
        knn_clf.fit(known_face_encodings, known_names)

        # Save the trained KNN classifier
        if MODEL_PATH is not None:
            with open(MODEL_PATH, 'wb') as f:
                pickle.dump(knn_clf, f)

        self.knownNames = known_names
        self.knownFaceEncodings = known_face_encodings

        print ("MODEL RETRAINED")
        return known_names, known_face_encodings


    def _predict(self, frame_number, frame, knn_clf, known_names, known_face_encodings):

        unknown_image = frame[:, :, ::-1]
        if max(unknown_image.shape) > 1600:
            pil_image = Image.fromarray(unknown_image)
            pil_image.thumbnail((1600,0), PIL.Image.LANCZOS)
            unknown_image = np.array(pil_img)


        img = cv2.cvtColor( unknown_image, cv2.COLOR_RGB2GRAY )
        pil_image = Image.fromarray(unknown_image)

        self.start_time = time.time()
        face_locations = face_recognition.face_locations(img, number_of_times_to_upsample=2, model="hog")
        print "Face detection--- %s seconds ---" % (time.time() - self.start_time)

        if not face_locations:
            return False

        if face_locations:
            print face_locations
            unknown_encodings = face_recognition.face_encodings(unknown_image, face_locations)

            print "amy", type(unknown_encodings[0])
            closest_distances = knn_clf.kneighbors(unknown_encodings, n_neighbors=1)

            matches = [closest_distances[0][i][0] <= DIST_THRESHOLD for i in range(len(face_locations))]

            return [(pred, loc, encoding, rec) if rec else ("unknown", loc, encoding, rec) for pred, encoding, loc, rec in zip(knn_clf.predict(unknown_encodings), unknown_encodings, face_locations, matches)]


    def _showPredictionLabelsOnImage(self, frame, data):

        # Draw a box around the face
        cv2.rectangle(frame, (data["left"], data["top"]), (data["right"], data["bottom"]), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (data["left"], data["bottom"] - 35), (data["right"], data["bottom"]), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, data["name"], (data["left"] + 6, data["bottom"] - 6), font, 1.0, (255, 255, 255), 1)

        return

    def detectFrameFromModel(self, image):

        frame = image
        result = {
            "faceFound" : False,
            "name": "",
        }

        frame_number = self.frameNumber
        self.frameNumber += 1

        if not len(self.knownNames) or not len(self.knownFaceEncodings):
            print "Did not find any previous data starting training model"
            self.scanKnownFaces()

        known_names = self.knownNames
        known_face_encodings = self.knownFaceEncodings

        with open(MODEL_PATH, 'rb') as f:
            knn_clf = pickle.load(f)

        predictions = self._predict(frame_number, frame, knn_clf, known_names, known_face_encodings)
        if(predictions == 0):
            print "No face detected"
        else:
            print "Face detected"
            print type(predictions)

            for name, (top, right, bottom, left),encoding,rec in predictions:
                array2 = 0
                for nam, k_encode in zip(known_names, known_face_encodings):
                    if(nam == name):
                        array2 = k_encode

                print "- Found {} at ({}, {}) frame no:".format(name, left, top), frame_number
                print "Face Recognition--- %s seconds ---" % (time.time() - self.start_time)

                data = {
                    "name": name,
                    "top": top,
                    "right": right,
                    "bottom": bottom,
                    "left": left
                }

                # add label on detections
                self._showPredictionLabelsOnImage(frame, data)

                if(name != "unknown"):
                    IMAGE_PATH = DETECTED_IMAGE_SAVE_PATH + '/' + str(datetime.date.today())
                    if not os.path.exists(IMAGE_PATH):
                        os.makedirs(IMAGE_PATH)
                    FaceFileName = IMAGE_PATH + "/{}_{}.jpg".format(name, datetime.datetime.now().strftime('%H:%M:%S'))
                    cv2.imwrite(FaceFileName, frame)

            result = {
                "faceFound" : True,
                "name": name
            }

        # Display the resulting image
        cv2.imshow('Video', frame)

        return result

    def run(self):
        known_names, known_face_encodings = self.scanKnownFaces()

        input_movie = cv2.VideoCapture(0)
        frame_number = 0

        while True:
            flag = 0
            ret, frame = input_movie.read()
            frame_number += 1

            # Quit when the input video file ends
            if not ret:
                break

            if(frame_number % 5 == 0):
                with open(MODEL_PATH, 'rb') as f:
                    knn_clf = pickle.load(f)

                predictions = self._predict(frame_number, frame, knn_clf, known_names, known_face_encodings)
                if(predictions == 0):
                    print "No face detected"
                else:
                    print "Face detected"
                    print type(predictions)
                    i = 0

                    for name, (top, right, bottom, left),encoding,rec in predictions:
                        array2 = 0
                        for nam, k_encode in zip(known_names, known_face_encodings):
                            if(nam == name):
                                array2 = k_encode
                            i += 1
                        print "- Found {} at ({}, {}) frame no:".format(name, left, top), frame_number
                        print "Face Recognition--- %s seconds ---" % (time.time() - self.start_time)

                        data = {
                            "name": name,
                            "top": top,
                            "right": right,
                            "bottom": bottom,
                            "left": left
                        }

                        # add label on detections
                        self._showPredictionLabelsOnImage(frame, data)

                        if(name != "unknown"):
                            FaceFileName = DETECTED_IMAGE_SAVE_PATH + "{}_{}_{}.jpg".format(name, i, frame_number)
                            cv2.imwrite(FaceFileName, frame)

                # Display the resulting image
                cv2.imshow('Video', frame)

                # Hit 'q' on the keyboard to quit!
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        return



if __name__ == "__main__":
    detector = FaceDetector()
    detector.run()
