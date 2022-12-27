import cv2
import mediapipe as mp


class HandDetector:

    def __init__(self, mode=False, max_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        # Setting all the values for Hand Detection
        self.mode = mode
        self.max_hands = max_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=mode, max_num_hands=self.max_hands,
                                        min_detection_confidence=self.min_detection_confidence,
                                        min_tracking_confidence=self.min_tracking_confidence)
        self.mpDraw = mp.solutions.drawing_utils

    def DrawHands(self, img, draw=True):
        if draw:
            # Converting the image from BGR to RGB as MediaPipe works on RGB images
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Getting the landmarks of the hand
            results = self.hands.process(imgRGB)

            if results.multi_hand_landmarks:
                for handLms in results.multi_hand_landmarks:
                    # Drawing all the landmarks and connections on the image
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

    def givePosition(self, img, draw=True):
        # Converting the image from BGR to RGB as MediaPipe works on RGB images
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Getting the landmarks of the hand
        results = self.hands.process(imgRGB)

        x = None
        y = None
        pos = []
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                for ID, LM in enumerate(handLms.landmark):
                    # Getting the size of the window
                    h, w, c = img.shape

                    # Converting the landmark position according to the size of the window
                    cx, cy = int(LM.x * w), int(LM.y * h)

                    # Appending the ID and the position of the landmarks to the list
                    pos.append([ID, cx, cy])
                    if draw:
                        cv2.circle(img, (cx, cy), 7, (255, 0, 255), cv2.FILLED)

        return pos
