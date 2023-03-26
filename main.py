import cv2
import numpy as np
from PIL import Image
import numpy
def AddImageToCenter(frame, x, y, img):
    pilim = Image.fromarray(frame)
    pilim.paste(img, box=(int(x) - 32, int(y) - 32), mask=img)
    frame = np.array(pilim)
    return frame

def PointSearch(frame, templateImage, sensitivity):
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    grayImage = cv2.cvtColor(img_label, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(grayFrame, grayImage, cv2.TM_CCOEFF_NORMED)
    threshold = sensitivity
    locate = np.where(res >= threshold)
    return locate

src = cv2.imread(r'variant-2.png', 1)
dst = cv2.GaussianBlur(src, (5, 5), cv2.BORDER_DEFAULT)
cv2.imshow("Gaussian Smoothing", numpy.hstack((src, dst)))
cv2.waitKey(0)
cv2.destroyAllWindows()
cap = cv2.VideoCapture(0)
label = "refPoint.jpg"
speedo = Image.open('fly64.png').convert('RGBA')
img_label = cv2.imread(label)
heightPoint, widthPoint, channels = img_label.shape

while True:
    ret, frame = cap.read()
    loc = PointSearch(frame, img_label, 0.6)
    # display of screen elements
    if len(loc[0]) > 0:
        x = loc[1][0]
        y = loc[0][0]
        cv2.rectangle(frame, (x, y), (x + widthPoint, y + heightPoint), (0, 255, 0), 2)

        cx = x + widthPoint / 2
        cy = y + heightPoint / 2
        distance_to_centerX = abs(frame.shape[1] / 2 - cx)
        distance_to_centerY = abs(frame.shape[0] / 2 - cy)

        text = "coordinate"
        textXY = "x: " + str( distance_to_centerX) + "   y: " + str( distance_to_centerY) + " pixels"
        cv2.putText(frame, text, (10, 435), cv2.FONT_HERSHEY_SIMPLEX, 1,  (0,0,255), 2)
        cv2.putText(frame, textXY, (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 1,  (0,0,255), 2)

        frame = AddImageToCenter(frame, cx,cy, speedo)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()