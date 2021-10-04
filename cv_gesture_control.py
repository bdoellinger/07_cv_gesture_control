import cv2 
import time
from cv_hand_tracking_module import handDetector
from math import hypot
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

################################--------------- Access webcam and volume control ---------------################################
print("-" * 100)
# try to get a working webcam, id == 0 should be internal (e.g. built-in webcam in laptops), id >= 1 should be external
for webcam_id in range(5):
    cap = cv2.VideoCapture(webcam_id) 
    if cap.isOpened():
        camera_recognised = True
        print(f"camera with id == {webcam_id} successfully recognised")
        break
else:
    camera_recognised = False
    print("default camera / external camera not recognised, please check if webcam is plugged in and working")


# 720p resolution
cam_width, cam_height = 1280, 720
cap.set(3, cam_width)
cap.set(4, cam_height)


# to access volume control of computer
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
min_Volume, max_Volume, _ = volume.GetVolumeRange()   # range is usually -96.0, 0.0 or -65.25, 0.0
Volume_range = max_Volume - min_Volume
#volume.SetMasterVolumeLevel(0.0, None)               # use this to set volume / scale is non-linear, 0.0 -> 100%, -1.0 -> 94%, -5.0 -> 72%, -10.0 -> 52%, -20.0 -> 27%, -40.0 -> 7% -65.25 -> 1%

# if distance of thumb and index finger is <= min_length we want to set volume to 0%, if distance >= max_length we set volume to 100%
min_length, max_length = 50, 200

################################--------------- Process image in real time ---------------################################
# create instance of hand detector class, play with detection confidence (0 <= detectionCon <= 1) 
# high value reduces flickering, but model is also less likely to detect hand / low value makes it easier to detect hand, but also can lead to false positives
hand_detector = handDetector(maxHands=2, detectionCon=0.75)


previous_time = time.time() # time at start to determine FPS
print("image processing started, wait a moment for image detection to initialize") if camera_recognised else print("image processing can't be started")

while camera_recognised:
    # get image from webcam
    camera_recognised, img = cap.read()


    # find and mark hand landmarks in image (by default maximum of 2 hands)
    hand_detector.findHands(img)


    # get list of hand landmarks (fingertips, knuckels, etc, ...) of first detected hand
    lmList = hand_detector.findLandmarkPositions(img, handNumber=0, draw=False)

    # determine volume according to distance between thumb and index finger, and visualize
    if len(lmList) > 8: # since we need landmarks with ids 4 (tip of thump), 8 (tip of index finger)
        # show line between thumb and index finger
        coordinates_thumb = lmList[4][1:]
        coordinates_index_finger = lmList[8][1:]
        mid_point = ((coordinates_thumb[0] + coordinates_index_finger[0]) // 2, (coordinates_thumb[1] + coordinates_index_finger[1]) // 2)

        cv2.circle(img, coordinates_thumb, 8, (255, 0, 0), cv2.FILLED)
        cv2.circle(img, coordinates_index_finger, 8, (255, 0, 0), cv2.FILLED)

        cv2.line(img, coordinates_thumb, coordinates_index_finger, (255, 0, 0), 3)
        cv2.circle(img, mid_point, 15, (255, 0, 0), cv2.FILLED)

        line_length = hypot(coordinates_thumb[0] - coordinates_index_finger[0], coordinates_thumb[1] - coordinates_index_finger[1])

        # indicate wether or not we are at minimum / maximum Volume (choose range 50 - 300 as min and max distance between thumb and index finger)
        if line_length <= min_length:
            cv2.circle(img, mid_point, 15, (0, 0, 255), cv2.FILLED)
        elif line_length >= max_length:
            cv2.circle(img, mid_point, 15, (0, 255, 0), cv2.FILLED)

        # adjust volume and draw volume bar
        # t indicates the percentage we traveled from min_length to max_length, t==0 implies length <= min_length, t==1 implies length >= max_length 
        # -> line_length = min_length + t * (max_length - min_length)
        t = (line_length - min_length) / (max_length - min_length)
        t = min(max(0, t), 1)   # such that 0 <= t <= 1
        new_Volume = min_Volume + t * Volume_range
        volume.SetMasterVolumeLevel(new_Volume, None)

        # draw bar indicating t (which corresponds to volume)
        cv2.rectangle(img, (10-1,80-1), (40+1, cam_height-20+1), (0, 0, 0), 2)
        cv2.rectangle(img, (10,80 + int((1-t)*(cam_height-100))), (40, cam_height-20), (0, int(t * 255), int((1-t) * 255)), -1)


    # calculate frames per second
    current_time = time.time()
    FPS = int(1 // (current_time - previous_time))
    previous_time = current_time
    cv2.putText(img, f"FPS: {FPS}", (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 3) # use this for white text (255, 255, 255)


    # display processed image
    window = cv2.imshow("Img", img)
    cv2.waitKey(1)


    # to close the window (by pressing x button on window or q on keyboard)
    if cv2.getWindowProperty("Img", 0) < 0 or (cv2.waitKey(1) & 0xFF == ord("q")):
        cv2.destroyAllWindows()
        break

print("image processing finished")
print("-" * 100)


################################--------------- Main Routine ---------------################################
def main():
    pass


if __name__ == "__main__":
    main()

