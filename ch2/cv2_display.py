#!/usr/bin/env python 
import cv2

original = cv2.imread('home_tree.png')
gray = cv2.imread('home_tree.png', 0)

cv2.imshow('Original', original)
cv2.imshow('Gray', gray)

cv2.waitKey(0)
cv2.destroyAllWindows()
