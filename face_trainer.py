import os
import cv2
import math
import time
import datetime
import numpy as np
from PIL import Image, ImageDraw
import face_recognition.api as face_recognition

####################################
# DEVELOPER        : VIKRAM SINGH  #
# TECHNOLOGY STACK : PYTHON        #
####################################


# =============================== CONFIG =======================================

RECOGNISED_NAMES         = "recorded_data/names.txt" # saved names list of people recognised
RECOGNISED_ENCODINGS     = "recorded_data/encodings.txt" # saved encoding list of people recognised
TRAINED_IMAGES           = "recorded_data/images/trained/"

# ==============================================================================


class FaceTrainer():

    def __init__(self):

        self.frameNumber         = 1
        self.start_time          = time.time()
        self.knownNames          = []


    def _trainImage(self, unknown_image, username="unknown"):
        with open(RECOGNISED_NAMES) as f:
            self.knownNames = f.read().splitlines()

        unknown_image = cv2.cvtColor( unknown_image, cv2.COLOR_BGR2RGB )
        img = cv2.cvtColor( unknown_image, cv2.COLOR_BGR2GRAY )

        face_locations = face_recognition.face_locations(unknown_image, number_of_times_to_upsample=2, model="hog")
        print "Locations : ", face_locations
        if not face_locations: print "No face detected"
        unknown_encodings = []

        if face_locations:
            print unknown_encodings
            unknown_encodings = face_recognition.face_encodings(unknown_image, face_locations)
            pil_image = Image.fromarray(unknown_image)
            draw = ImageDraw.Draw(pil_image)
            for (top, right, bottom, left), face_encoding in zip(face_locations, unknown_encodings):
                encodingsfile = open(RECOGNISED_ENCODINGS, 'a+')
                for x in face_encoding:
                    encodingsfile.write("%f " % x)
                encodingsfile.write("\n")

                # making output annotation for face
                draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))
                text_width, text_height = draw.textsize(username)
                draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
                draw.text((left + 6, bottom - text_height - 5), username, fill=(255, 255, 255, 255))

                self.knownNames.append(username)
                namefile = open(RECOGNISED_NAMES , 'w')
                for item in self.knownNames:
                    namefile.write("%s\n" % item)
                namefile.close()
            del draw

            # If one wants to see image at runtime
            # pil_image.show()
            return (True, pil_image)

        return (False, None)


    def trainForBot(self, name, frame):
        res = self._trainImage(frame, name)
        if res[0]:
            fileName = TRAINED_IMAGES + str(name) + str(self.frameNumber) + '.png'
            res[1].save(fileName, 'PNG')
            self.frameNumber += 1

        return (res[0], res[1])


    def run(self):
        username = raw_input("Enter name of Unknown : ")
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            cv2.imshow("trial", frame)

            # Hit 'Esc' on the keyboard while mouse on video frame to capture image and exit!
            if cv2.waitKey(1) == 27:

                res = self._trainImage(frame, username)
                # cv2.imwrite('images/{}.jpg'.format(username), frame)
                break

        cv2.destroyAllWindows()
        cap.release()
        return


if __name__ == "__main__":
    trainer = FaceTrainer()
    trainer.run()
