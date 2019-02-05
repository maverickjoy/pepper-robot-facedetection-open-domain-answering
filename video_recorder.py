import qi
import argparse
import sys
import time
from PIL import Image
import requests
import json
import base64
import random
import numpy as np
import cv2
from basic_awareness import HumanTrackedEventWatcher

# ==============================================================================
#                      --- CAMERA INFORMATION ---

# AL_resolution
AL_kQQQQVGA = 8     #Image of 40*30px
AL_kQQQVGA  = 7     #Image of 80*60px
AL_kQQVGA   = 0     #Image of 160*120px
AL_kQVGA    = 1     #Image of 320*240px
AL_kVGA     = 2     #Image of 640*480px
AL_k4VGA    = 3     #Image of 1280*960px
AL_k16VGA   = 4     #Image of 2560*1920px

# Camera IDs
AL_kTopCamera    = 0
AL_kBottomCamera = 1
AL_kDepthCamera  = 2

# Need to add All color space variables
AL_kRGBColorSpace =    13


# ==============================================================================


class VideoRecorder(object):

    def __init__(self, app, human_tracked_event_watcher):
        super(VideoRecorder, self).__init__()

        try:
            app.start()
        except RuntimeError:
            print ("Can't connect to Naoqi at ip \"" + args.ip + "\" on port " +
                   str(args.port) + ".\n")

            sys.exit(1)

        session = app.session

        # SUBSCRIBING SERVICES
        self.motion            = session.service("ALMotion")
        self.posture_service   = session.service("ALRobotPosture")
        self.video             = session.service("ALVideoDevice")
        self.human_watcher     = human_tracked_event_watcher

        # INITIALISING CAMERA POINTERS
        self.imageNo2d = 1
        self.imageNo3d = 1

    def _retrieveVideoFeed(self):
        # Capture Image in RGB

        # Top Camera
        cameraId = 0
        # WARNING : The same Name could be used only six time.
        strName = "capture2DImage_{}".format(random.randint(1,10000000000))
        clientRGB = self.video.subscribeCamera(strName, cameraId, AL_kQVGA, AL_kRGBColorSpace, 10)

        # create image
        width = 320
        height = 240
        image = np.zeros((height, width, 3), np.uint8)

        while True:
            print "Starting Video Feed to cancel press ESC"

            # get image
            result = self.video.getImageRemote(clientRGB);

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
                cv2.imshow("pepper-top-camera-640*480px", image)

            # exit by [ESC]
            if cv2.waitKey(33) == 27:
                break


    def run(self):
        # start
        print "Waiting for the robot to be in wake up position"
        self.motion.wakeUp()
        self.posture_service.goToPosture("StandInit", 0.1)
        time.sleep(3)
        self.human_watcher.start_basic_awareness()

        print "Starting"
        self._retrieveVideoFeed()

        print "Waiting for the robot to be in rest position"
        # self.motion.rest()




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
    app = qi.Application(["VideoRecorder",
                          "--qi-url=" + connection_url])

    human_tracked_event_watcher = HumanTrackedEventWatcher(app)
    event_watcher = VideoRecorder(app, human_tracked_event_watcher)
    event_watcher.run()
