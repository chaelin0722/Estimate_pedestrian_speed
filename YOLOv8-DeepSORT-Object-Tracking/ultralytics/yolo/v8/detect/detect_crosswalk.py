import cv2
import numpy as np


def cross(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #blurred = cv2.GaussianBlur(gray,(15, 15), 6)
    ret, thresh = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


    for c in contours:
        # if the contour is not sufficiently large, ignore it
        area = cv2.contourArea(c)

        # Fill very small contours with zero (erase small contours).
        if area < 3000:
            cv2.fillPoly(thresh, pts=[c], color=0)
            continue
        # get the min area rect
        rect = cv2.minAreaRect(c)

        (x, y), (w, h), angle = rect
        aspect_ratio = max(w, h) / min(w, h)

        # Assume zebra line must be long and narrow (long part must be at lease 1.5 times the narrow part).
        if (aspect_ratio < 10):
            cv2.fillPoly(thresh, pts=[c], color=0)
            continue

    #########
    # 가까운 컨투어 합치기
    thresh_gray = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (100,100)));

    # Find contours in thresh_gray after closing the gaps
    contours, _ = cv2.findContours(thresh_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for c in contours:
        area = cv2.contourArea(c)

        # Small contours are ignored.
        if area < 2000:
            cv2.fillPoly(thresh, pts=[c], color=0)
            continue

        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        # convert all coordinates floating point values to int
        box = np.int0(box)
        cv2.drawContours(img, [box], 0, (0, 0, 255), 1)



    return img

