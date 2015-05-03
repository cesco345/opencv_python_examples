import cv2
import numpy as np

def thresh_callback(thresh):
    edges = cv2.Canny(blur,thresh,thresh*2)
    drawing = np.zeros(img.shape,np.uint8)                  # Image to draw the contours
    contours,hierarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        moments = cv2.moments(cnt)                          # Calculate moments
        if moments['m00']!=0:
            cx = int(moments['m10']/moments['m00'])         # cx = M10/M00
            cy = int(moments['m01']/moments['m00'])         # cy = M01/M00
            moment_area = moments['m00']                    # Contour area from moment
            contour_area = cv2.contourArea(cnt)             # Contour area using in_built function
            
            cv2.drawContours(drawing,[cnt],0,(0,255,0),1)   # draw contours in green color
            cv2.circle(drawing,(cx,cy),5,(0,0,255),-1)      # draw centroids in red color
    cv2.imshow('output',drawing)
    cv2.imshow('input',img)

img = cv2.imread('plate.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(5,5),0)

cv2.namedWindow('input')

thresh = 200
max_thresh = 255

cv2.createTrackbar('canny thresh:','input',thresh,max_thresh,thresh_callback)

thresh_callback(200)

if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()
