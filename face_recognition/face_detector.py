# -*- coding: utf-8 -*-
from __future__ import print_function
import click
import os
import math
import face_recognition.api as face_recognition
import cv2,dlib
import numpy as np
import pickle
import time
from PIL import Image, ImageDraw
from sklearn import neighbors
import dlib.cuda as cuda
start_time = time.time()
distance_threshold=0.45


model_save_path = "models/model1.txt"

#Retrain model for any new entries of faces
def scan_known_people():
    with open('./names.txt') as f:
        known_names = f.read().splitlines()

    known_face_encodings=np.loadtxt('encodings.txt',dtype='float')

    n_neighbors = int(round(math.sqrt(len(known_face_encodings))))
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors, algorithm='ball_tree', weights='distance')
    print(known_face_encodings)
    knn_clf.fit(known_face_encodings, known_names)

    # Save the trained KNN classifier
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)

    print ("MODEL RETRAINED")
    return known_names, known_face_encodings



def predict(frame_number,frame,knn_clf,known_names, known_face_encodings):

    unknown_image = frame[:, :, ::-1]
    if max(unknown_image.shape) > 1600:
        pil_image = Image.fromarray(unknown_image)
        pil_image.thumbnail((1600,0), PIL.Image.LANCZOS)
        unknown_image = np.array(pil_img)


    img = cv2.cvtColor( unknown_image, cv2.COLOR_RGB2GRAY )
    pil_image =Image.fromarray(unknown_image)
    # pil_image.show()
    #Find locations of faces in image
    start_time = time.time()
    face_locations = face_recognition.face_locations(img,number_of_times_to_upsample=2, model="hog")
    print("Face detection--- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    if not face_locations:
        return 0
    if face_locations:
        print(face_locations)
        unknown_encodings = face_recognition.face_encodings(unknown_image,face_locations)
        #print(type(unknown_encodings))
        print("amy",type(unknown_encodings[0]))
        closest_distances = knn_clf.kneighbors(unknown_encodings, n_neighbors=1)

        matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(face_locations))]

        i=0
        return [(pred, loc,encoding,rec) if rec else ("unknown", loc,encoding,rec) for pred,encoding, loc, rec in zip(knn_clf.predict(unknown_encodings),unknown_encodings, face_locations,matches)]




def show_prediction_labels_on_image(frame, data):

    # Draw a box around the face
    cv2.rectangle(frame, (data["left"], data["top"]), (data["right"], data["bottom"]), (0, 0, 255), 2)

    # Draw a label with a name below the face
    cv2.rectangle(frame, (data["left"], data["bottom"] - 35), (data["right"], data["bottom"]), (0, 0, 255), cv2.FILLED)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(frame, data["name"], (data["left"] + 6, data["bottom"] - 6), font, 1.0, (255, 255, 255), 1)

    return


@click.command()
@click.option('--cpus', default=1, help='number of CPU cores to use in parallel (can speed up processing lots of images). -1 means "use all in system"')
@click.option('--tolerance', default=0.5, help='Tolerance for face comparisons. Default is 0.6. Lower this if you get multiple matches for the same person.')
@click.option('--show-distance', default=False, type=bool, help='Output face distance. Useful for tweaking tolerance setting.')
def main( cpus, tolerance, show_distance):
    known_names, known_face_encodings = scan_known_people()

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
            with open(model_save_path, 'rb') as f:
                knn_clf = pickle.load(f)

            predictions=predict(frame_number,frame,knn_clf,known_names, known_face_encodings)
            if(predictions==0):
                print("No face detected")

            else:
                print("Face detected")
                print(type(predictions))
                i=0

                for name, (top, right, bottom, left),encoding,rec in predictions:
                    array2=0
                    for nam,k_encode in zip(known_names, known_face_encodings):
                        if(nam==name):
                            array2=k_encode
                        i=i+1
                    print("- Found {} at ({}, {}) frame no:".format(name, left, top),frame_number)
                    print("Face Recognition--- %s seconds ---" % (time.time() - start_time))
                    #similarity=1-spatial.distance.cosine(encoding,array2)
                    #print("probability",similarity)

                    data = {
                        "name": name,
                        "top": top,
                        "right": right,
                        "bottom": bottom,
                        "left": left
                    }
                    show_prediction_labels_on_image(frame, data)

                    if(name != "unknown"):
                        FaceFileName = "detections/" + name+"_"+str(i)+"_"+str(frame_number) + ".jpg"
                        # cv2.imwrite(FaceFileName, frame)

            # Display the resulting image
            cv2.imshow('Video', frame)

            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break



if __name__ == "__main__":
    main()
