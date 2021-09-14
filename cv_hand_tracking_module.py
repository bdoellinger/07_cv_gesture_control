import cv2
import mediapipe as mp


################################--------------- Hand Detection Class ---------------################################
class handDetector():

    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode, max_num_hands=self.maxHands, min_detection_confidence=self.detectionCon, min_tracking_confidence=self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils


    def findHands(self, img, draw=True, draw_landmarks=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # convert to RGB, since hands only uses RGB images
        results = self.hands.process(imgRGB)
        
        if results.multi_hand_landmarks:
        
            for handLms in results.multi_hand_landmarks:
                # draw hand landmarks into original image (which will be displayed), unless explicitly stated not to (draw == False)
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)  

                if draw_landmarks:
                    for lm in handLms.landmark:
                        h, w, c = img.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(img, (cx, cy), 5, (255, 255, 0), cv2.FILLED)
        
        return img


    def findLandmarkPositions(self, img, handNumber=0, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # convert to RGB, since hands only uses RGB images
        results = self.hands.process(imgRGB)

        lmList = []

        if results.multi_hand_landmarks:
            
            # in case we have not detected enough hands
            if handNumber >= len(results.multi_hand_landmarks):
                return lmList

            handLms = results.multi_hand_landmarks[handNumber]
            
            for idx, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append((idx, cx, cy))
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        return lmList


################################--------------- Main Routine ---------------################################
def main():
    pass


if __name__ == "__main__":
    main()

