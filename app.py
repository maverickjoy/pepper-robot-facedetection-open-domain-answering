from opendomain_responder import basic_qa
import qi
import argparse
import os
import sys
import time
from PIL import Image
import requests
import json
import base64
import random
import numpy as np
import cv2
import datetime
from face_trainer import FaceTrainer
from face_detector import FaceDetector
from basic_awareness import HumanTrackedEventWatcher


####################################
# DEVELOPER        : VIKRAM SINGH  #
# TECHNOLOGY STACK : PYTHON        #
####################################


# ==============================================================================
#                      --- CAMERA INFORMATION ---

# AL_resolution
AL_kQQQQVGA       = 8 # Image of 40*30px
AL_kQQQVGA        = 7 # Image of 80*60px
AL_kQQVGA         = 0 # Image of 160*120px
AL_kQVGA          = 1 # Image of 320*240px
AL_kVGA           = 2 # Image of 640*480px
AL_k4VGA          = 3 # Image of 1280*960px
AL_k16VGA         = 4 # Image of 2560*1920px

# Camera IDs
AL_kTopCamera     = 0
AL_kBottomCamera  = 1
AL_kDepthCamera   = 2

# Need to add All color space variables
AL_kRGBColorSpace = 13

# ==============================================================================

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# =============================== CONFIG =======================================

NODE_IP                                     = "10.9.43.90" # IP Address of the node in which you are running this program

# Name of topic file situated at robots dir => `/home/nao/chat/`
TOPIC_NAME                                  = "face_detect_greeter.top"
GREET_NAMES_LOGS_PATH                       = "recorded_data/greet_logs/" # saved names list of people recognised
TIME_GAP_TO_WISH_GUEST                      = 5 # In minutes
TIME_GAP_TO_WISH_KNOWN_PERSON               = 5 # In minutes

# ==============================================================================


# =============== ORIENTATION BASED ON INITIAL BOT POSITION ====================
PHI = 0  # AMOUNT TO MOVE TO FACE TOWARDS EAST (0`)
#               0
#   math.pi/2       -math.pi/2
#            math.pi
PENDING_PHI = 0
# ==============================================================================


class FaceRecognitionGreeter(object):

    def __init__(self, app, human_tracked_event_watcher, face_detector, face_trainer):
        super(FaceRecognitionGreeter, self).__init__()

        try:
            app.start()
        except RuntimeError:
            print ("Can't connect to Naoqi at ip \"" + args.ip + "\" on port " +
                   str(args.port) + ".\n")

            sys.exit(1)

        session = app.session
        self.subscribers_list = []

        # SUBSCRIBING SERVICES
        self.memory_service    = session.service("ALMemory")
        self.motion_service    = session.service("ALMotion")
        self.posture_service   = session.service("ALRobotPosture")
        self.video_service     = session.service("ALVideoDevice")
        self.tts               = session.service("ALTextToSpeech")
        self.speaking_movement = session.service("ALSpeakingMovement")
        self.dialog_service    = session.service("ALDialog")
        self.tablet_service    = session.service("ALTabletService")
        self.human_watcher     = human_tracked_event_watcher
        self.face_trainer      = face_trainer
        self.face_detector     = face_detector

        # INITIALISING Variables
        self.imageNo2d          = 1
        self.lastGuestTimestamp = 0
        self.startFaceTraining  = True
        self.userName           = '-'

    def create_callbacks(self):

        self.connect_callback("developer_name_event",
                              self.developer_name_event)

        # self.connect_callback("Dialog/LastInput",
        #                       self.last_dialog_event)

        self.connect_callback("unknown_input_event",
                              self.who_question_event)

        self.connect_callback("specific_question_event",
                              self.specific_question_event)

        self.connect_callback("register_human_event",
                              self.register_human_event)

        self.tablet_service.onInputText.connect(self.got_button_event)


        return

    def connect_callback(self, event_name, callback_func):
        print "Callback connection"
        subscriber = self.memory_service.subscriber(event_name)
        subscriber.signal.connect(callback_func)
        self.subscribers_list.append(subscriber)

        return

    def got_button_event(self, buttonId, value):
        if buttonId == 1:
            self.userName = value
            if value == '':
                self.userName = "NULL"
        if buttonId == 0: # Cancel is pressed
            self.userName = "NULL"
        self._clearTablet()
        return

    def register_human_event(self, value):
        print "Registering face training event"
        self.startFaceTraining = True
        self._makePepperSpeak("Please start looking into my eyes untill I say done")
        time.sleep(1)
        return


    def developer_name_event(self, value):
        # setting Memory Variable
        self.memory_service.insertData("developerName", "Vikram Singh")
        return

    # def last_dialog_event(self, value):
    #     print "Last Dialog Event Called"
    #     print "Value : ", value
    #     return

    def who_question_event(self, value):
        print "Question : ", value
        ans = basic_qa.answerQues(value)
        if ans:
            print "Answer : ", ans
            self._makePepperSpeak(ans)
        return


    def specific_question_event(self, value):
        print "Value : ", value
        msg = ''
        if value == 'TIME':
            msg = "Present time is " + time.ctime().split()[3]

        if value == 'DAY':
            msg = "The day today is " + datetime.datetime.today().strftime('%A')

        if value == 'DATE':
            msg = "The date today is " + datetime.datetime.today().strftime('%D')

        if value == 'MONTH':
            msg = "The present month is " + datetime.datetime.today().strftime('%B')

        if value == 'YEAR':
            msg = "The present year is " + time.ctime().split()[4]

        if msg:
            self._makePepperSpeak(msg)

        return


    def _makePepperSpeak(self, userMsg):
        # MAKING PEPPER SPEAK
        # future = self.animation_player_service.run("animations/Stand/Gestures/Give_3", _async=True)
        sentence = "\RSPD=" + str(100) + "\ "  # Speed
        sentence += "\VCT=" + str(100) + "\ "  # Voice Shaping
        sentence += userMsg
        sentence += "\RST\ "
        self.tts.say(str(sentence))
        # future.value()

        return

    def _startFaceTrainProcedure(self, image):
        print
        print
        print

        name = ""
        # Capture name from tablet
        self._makePepperSpeak("Please enter your name")
        self.tablet_service.showInputTextDialog("Please Enter Your Name", "Okay", "Cancel")

        while True:
            if self.userName != '-':
                if self.userName == "NULL":
                    self.userName = '-'
                    print "Cancel button clicked or name not given, hence cannot training image"
                    self.startFaceTraining = False
                    return
                else:
                    name = self.userName
                    self.userName = '-'
                    break

        res = self.face_trainer.trainForBot(name, image)
        if res[0]:
            self._makePepperSpeak("Your Face Training is done")
            # Saving file temporarily for displaying
            base_url = "/var/www/html/pepper_hack/images/" # your localhost server addr
            view_url = base_url + str(name) + '.png'
            res[1].save(view_url, 'PNG')
            self._showOnTablet(str(name) + '.png', False)
            time.sleep(5)
            # Deleting File
            os.remove(view_url)
            self._clearTablet()
        else:
            self._makePepperSpeak("Sorry I couldn't register you at the moment please try after some time")

        self.startFaceTraining = False
        return

    def _startFrameDetect(self, image):
        print
        print
        print
        data = self.face_detector.detectFrameFromModel(image)
        if data["faceFound"]:
            if data["name"] == "unknown":
                if (time.time() - self.lastGuestTimestamp) / 60 > TIME_GAP_TO_WISH_GUEST:
                    self.lastGuestTimestamp = time.time()
                    self._makePepperSpeak("Hello Guest")
            else:
                # Log the greeting
                LOG_FILE_PATH = GREET_NAMES_LOGS_PATH + \
                    '/' + str(datetime.date.today()) + '.log'
                with open(LOG_FILE_PATH, 'a+') as f:
                    userFound = False
                    lastTimestamp = 0
                    lines = f.readlines()
                    for line in lines:
                        username = line.split(',')[0].strip()
                        tStmp = float(line.split(',')[2].strip())
                        if str(username).lower() == str(data["name"]).lower():
                            userFound = True
                            lastTimestamp = tStmp
                    if userFound:
                        print "User already greeted"

                    if not userFound or (time.time() - lastTimestamp) / 60 > TIME_GAP_TO_WISH_KNOWN_PERSON:
                        f.write('{}, {}, {}\n'.format(
                            data["name"], time.ctime(), time.time()))
                        self._makePepperSpeak(
                            "Hello {} How are you".format(data["name"]))
        return

    def _retrieveVideoFeed(self):
        # Initializing the Detection Model
        self.face_detector.scanKnownFaces()

        # Top Camera
        cameraId = 0
        # WARNING : The same Name could be used only six time.
        strName = "capture2DImage_{}".format(random.randint(1, 10000000000))
        clientRGB = self.video_service.subscribeCamera(
            strName, cameraId, AL_kQVGA, AL_kRGBColorSpace, 10)

        # create image
        width = 320
        height = 240
        image = np.zeros((height, width, 3), np.uint8)

        while True:
            print "Starting Video Feed to cancel press ESC"

            # get image
            result = self.video_service.getImageRemote(clientRGB)

            if result == None:
                print 'cannot capture.'
            elif result[6] == None:
                print 'no image data string.'
            else:

                # translate value to mat
                values = map(ord, list(str(bytearray(result[6]))))
                i = 0
                for y in range(0, height):
                    for x in range(0, width):
                        image.itemset((y, x, 0), values[i + 0])
                        image.itemset((y, x, 1), values[i + 1])
                        image.itemset((y, x, 2), values[i + 2])
                        i += 3

                # show image
                cv2.imshow(
                    "pepper-top-camera-{}*{}".format(width, height), image)

                if self.startFaceTraining:
                    self._startFaceTrainProcedure(image)
                else:
                    self._startFrameDetect(image)

            # exit by [ESC]
            if cv2.waitKey(33) == 27:
                break

    def _clearTablet(self):
        # Hide the web view
        self.tablet_service.hideImage()
        self.tablet_service.hideDialog()

        return

    def _showOnTablet(self, mediaName, videoPresent):
        # Display Image On Tablet
        base_url = "http://{}/pepper_hack/images/".format(NODE_IP)
        view_url = base_url + str(mediaName)
        try:
            if videoPresent:
                self.tablet_service.playVideo(view_url)
            else:
                # self.tablet_service.showImageNoCache(view_url) # wont use this because we have toss images which are constant here
                self.tablet_service.showImage(view_url)

            print "Displaying on tablet : " + view_url
        except Exception, err:
            print "Error Showing On Tablet : " + str(err)

        return


    def _addTopic(self):
        print "Starting topic adding process"

        # Controlling hand gestures and movement while speaking
        self.speaking_movement.setEnabled(True)

        self.dialog_service.setLanguage("English")
        # Loading the topic given by the user (absolute path is required)

        topic_path = "/home/nao/chat/{}".format(TOPIC_NAME)

        topf_path = topic_path.decode('utf-8')
        self.topic_name = self.dialog_service.loadTopic(
            topf_path.encode('utf-8'))

        # Activating the loaded topic
        self.dialog_service.activateTopic(self.topic_name)

        # Starting the dialog engine - we need to type an arbitrary string as the identifier
        # We subscribe only ONCE, regardless of the number of topics we have activated
        self.dialog_service.subscribe('face_detector_example')

        print "\nSpeak to the robot using rules. Robot is ready"

        return

    def _cleanUp(self):
        print "Starting Clean Up process"
        self.human_watcher.stop_basic_awareness()

        # Stopping any movement if there
        self.motion_service.stopMove()
        # stopping the dialog engine
        self.dialog_service.unsubscribe('face_detector_example')
        # Deactivating the topic
        self.dialog_service.deactivateTopic(self.topic_name)

        # now that the dialog engine is stopped and there are no more activated topics,
        # we can unload our topic and free the associated memory
        self.dialog_service.unloadTopic(self.topic_name)

        self._clearTablet()
        self.posture_service.goToPosture("StandInit", 0.1)

        return


    def startGreeterProgram(self):
        print "Starting Greeter"
        self._retrieveVideoFeed()
        return

    def run(self):
        # start
        print "Waiting for the robot to be in wake up position"
        self.motion_service.wakeUp()
        self.posture_service.goToPosture("StandInit", 0.1)
        time.sleep(1)

        self.create_callbacks()
        self._addTopic()

        self.human_watcher.start_basic_awareness()

        try:
            self.startGreeterProgram()
        except KeyboardInterrupt:
            print "Interrupted by user, shutting down"
            self._cleanUp()
            print "Waiting for the robot to be in rest position"
            self.motion_service.rest()
            sys.exit(0)

        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="10.9.45.11",
                        help="Robot IP address. On robot or Local Naoqi: use \
                        '127.0.0.1'.")
    parser.add_argument("--port", type=int, default=9559,
                        help="Naoqi port number")

    args = parser.parse_args()

    # Initialize qi framework.
    connection_url = "tcp://" + args.ip + ":" + str(args.port)
    app = qi.Application(["FaceRecognitionGreeter",
                          "--qi-url=" + connection_url])

    human_tracked_event_watcher = HumanTrackedEventWatcher(app)
    face_trainer = FaceTrainer()
    face_detector = FaceDetector()
    event_watcher = FaceRecognitionGreeter(
        app, human_tracked_event_watcher, face_detector, face_trainer)
    event_watcher.run()
