import cv2
from math import atan, pi, ceil

img  =  cv2.imread('plate.png', 0)
h,w = img.shape[:2]
cv2.imshow('Input Image', img)
m = cv2.moments(img)


x = m['m10']/m['m00']
y = m['m01']/m['m00']
mu02 = m['mu02']
mu20 = m['mu20']
mu11 = m['mu11']

lambda1 = 0.5*( mu20 + mu02 ) + 0.5*( mu20**2 + mu02**2 - 2*mu20*mu02 + 4*mu11**2 )**0.5
lambda2 = 0.5*( mu20 + mu02 ) - 0.5*( mu20**2 + mu02**2 - 2*mu20*mu02 + 4*mu11**2 )**0.5 
lambda_m = max(lambda1, lambda2)

angle =  ceil(atan((lambda_m - mu20)/mu11)*18000/pi)/100
print " "
print "The angle is ", angle, " degrees."
print ""

center = tuple(map(int, (x, y)))
rotmat = cv2.getRotationMatrix2D(center, angle , 1)
rotatedImg = cv2.warpAffine(img, rotmat, (w, h), flags = cv2.INTER_CUBIC)       
   
cv2.imshow('Rotated Image', rotatedImg)
cv2.imwrite('rotated.png', rotatedImg)
cv2.waitKey()
