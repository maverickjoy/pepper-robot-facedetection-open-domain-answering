from __future__ import print_function
import os
import re
import face_recognition.api as face_recognition
import multiprocessing
import itertools
import sys
import cv2,dlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
# import openface
import time

start_time = time.time()


def test_image( unknown_image):
    with open('./names.txt') as f:
        known_names = f.read().splitlines()

    known_face_encodings=np.loadtxt('encodings.txt',dtype='float')
    unknown_image = cv2.cvtColor( unknown_image, cv2.COLOR_BGR2RGB )
    img = cv2.cvtColor( unknown_image, cv2.COLOR_BGR2GRAY )
    print("Image location start")
    #Find locations of faces in image
    face_locations = face_recognition.face_locations(unknown_image,number_of_times_to_upsample=2, model="hog")
    print("Face detection--- %s seconds ---" % (time.time() - start_time))
    print("locations",face_locations)
    if not face_locations: print("No face detected")
    unknown_encodings=[]
    username = "unknown"

    if face_locations:
        # for i, face_rect in enumerate(face_locations):
        #     top, right, bottom,left = face_rect
        #     d = dlib.rectangle(left,top, right, bottom )
        #     alignedFace=face_aligner.align(200, unknown_image,d, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        #     alignedFace=cv2.cvtColor( alignedFace, cv2.COLOR_BGR2RGB )
        #     cv2.imwrite("alignedfaces/aligned_face__{}_.jpg".format(i), alignedFace)
        #     unknown_encoding = face_recognition.face_encodings(alignedFace)
        #     unknown_encodings.extend(unknown_encoding)
        print(unknown_encodings)
        unknown_encodings = face_recognition.face_encodings(unknown_image,face_locations)
        pil_image = Image.fromarray(unknown_image)
        draw = ImageDraw.Draw(pil_image)
        count=0
        for (top, right, bottom, left), face_encoding in zip(face_locations, unknown_encodings):
            encodingsfile = open('encodings.txt', 'a+')
            print((face_encoding))
            for x in face_encoding:
                encodingsfile.write("%f " % x)
            encodingsfile.write("\n")
            username = raw_input("Enter name of Unknown : ")

            # making output annotation for face
            draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))
            text_width, text_height = draw.textsize(username)
            draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
            draw.text((left + 6, bottom - text_height - 5), username, fill=(255, 255, 255, 255))

            known_names.append(username)
            namefile = open('names.txt', 'w')
            for item in known_names:
                namefile.write("%s\n" % item)
            namefile.close()
        del draw
        pil_image.show()
        return username

def main():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        cv2.imshow("trial", frame)
        if cv2.waitKey(1) == 27:
            name = test_image(frame)
            # cv2.imwrite('images/{}.jpg'.format(name), frame)
            break

    cv2.destroyAllWindows()
    cap.release()



if __name__ == "__main__":
    main()
