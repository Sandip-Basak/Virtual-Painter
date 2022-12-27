import cv2
import time
import HandsTrackingModule as hand
import numpy as np


def main():
    cap = cv2.VideoCapture(1)

    # Creating a canvas for drawing
    imgCanvas = np.zeros((720, 1280, 3), np.uint8)

    detector = hand.HandDetector(min_detection_confidence=0.9, max_hands=1)
    pTime = 0

    # List of modes and items in the menu
    ModeList = ["Draw", "Selection", "Clear", "None"]
    selectList = [0, 1, 2, 3, 4, 5, 6]

    # Holds the current mode and selected item
    mode = ModeList[0]
    select = selectList[0]

    # Holds the positions of the items in the menu
    selectBorder = [[(20, 20), (95, 95)], [(195, 20), (270, 95)], [(370, 20), (445, 95)], [(545, 20), (615, 95)],
                    [(715, 20), (790, 95)], [(890, 20), (965, 95)], [(1010, 10), (1140, 105)]]

    # Holds the BGR color code of the items
    bgrList = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 0, 255), (0, 255, 255), (255, 255, 0), (0, 0, 0)]

    # Thickness for Brush and Eraser
    brushThickness = 15
    eraserThickness = 30

    # Holds position of the previous x and y coordinate of index finger
    px, py = 0, 0

    while True:
        success, img = cap.read()

        # Inverting the image
        img = cv2.flip(img, 360)

        # Resizing the image to 1280 X 720
        img = cv2.resize(img, (1280, 720))

        # Getting landmarks
        pos = detector.givePosition(img=img, draw=False)

        # Copy of the image
        overlay = img.copy()  # Copy Image for Opacity
        opacity = 0.4
        cv2.rectangle(img, (0, 0), (1280, 115), (0, 0, 0), cv2.FILLED)
        img = cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0)  # Blending the rectangle with the original image

        if len(pos) != 0:

            # Determining the mode
            if pos[8][2] < pos[6][2] and pos[12][2] < pos[10][2] and pos[16][2] > pos[14][2] and pos[20][2] > pos[18][2]:
                mode = ModeList[1]
            elif pos[8][2] < pos[6][2] and pos[12][2] > pos[10][2] and pos[16][2] > pos[14][2] and pos[20][2] > pos[18][2]:
                mode = ModeList[0]
            else:
                mode = ModeList[3]

            # Getting the x, y coordinates of index finger
            x, y = pos[8][1:]

            # Checking Clear All button pressed
            if mode == ModeList[1] and y < 115 and 1150 < x < 1280:
                mode = ModeList[2]

            # Getting the item selected from above menu
            if mode == ModeList[1]:
                px, py = 0, 0
                if y < 115:
                    if 0 < x < 120:
                        select = selectList[0]
                    elif 175 < x < 290:
                        select = selectList[1]
                    elif 350 < x < 465:
                        select = selectList[2]
                    elif 525 < x < 635:
                        select = selectList[3]
                    elif 695 < x < 810:
                        select = selectList[4]
                    elif 870 < x < 985:
                        select = selectList[5]
                    elif 1045 < x < 1280:
                        select = selectList[6]

            # Drawing on the canvas
            if mode == ModeList[0]:
                if px == 0 and py == 0:
                    px, py = x, y

                if select == selectList[6]:
                    cv2.line(img, (px, py), (x, y), bgrList[select], eraserThickness)
                    cv2.line(imgCanvas, (px, py), (x, y), bgrList[select], eraserThickness)
                else:
                    cv2.line(img, (px, py), (x, y), bgrList[select], brushThickness)
                    cv2.line(imgCanvas, (px, py), (x, y), bgrList[select], brushThickness)
                px, py = x, y
            elif mode == ModeList[2]:
                # Resetting the canvas
                imgCanvas = np.zeros((720, 1280, 3), np.uint8)

            # Marks the landmark number 8
            cv2.circle(img, (x, y), brushThickness, bgrList[select], cv2.FILLED)

        # Drawing Layout
        cv2.rectangle(img, (20, 20), (95, 95), (0, 0, 255), cv2.FILLED)
        cv2.rectangle(img, (195, 20), (270, 95), (0, 255, 0), cv2.FILLED)
        cv2.rectangle(img, (370, 20), (445, 95), (255, 0, 0), cv2.FILLED)
        cv2.rectangle(img, (545, 20), (615, 95), (255, 0, 255), cv2.FILLED)
        cv2.rectangle(img, (715, 20), (790, 95), (0, 255, 255), cv2.FILLED)
        cv2.rectangle(img, (890, 20), (965, 95), (255, 255, 0), cv2.FILLED)
        cv2.rectangle(img, (1010, 10), (1140, 105), (255, 255, 255), cv2.FILLED)
        cv2.putText(img, "Eraser", (1025, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
        cv2.rectangle(img, (1150, 10), (1280, 105), (255, 255, 255), cv2.FILLED)
        cv2.putText(img, "CLS", (1180, 70), cv2.FONT_HERSHEY_PLAIN, 2.5, (0, 0, 0), 3)

        # Drawing Selected border
        cv2.rectangle(img, selectBorder[select][0], selectBorder[select][1], (0, 0, 0), 3)

        # Calculating FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f"FPS-{int(fps)}", (1180, 710), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 1)

        # Merging the two images img and imgCanvas
        imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img, imgInv)
        img = cv2.bitwise_or(img, imgCanvas)

        # Showing the Image
        cv2.imshow("Painter", img)

        # On 'q' press the loop breaks
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # After the loop release the cap object
    cap.release()
    # Destroy all the windows
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
