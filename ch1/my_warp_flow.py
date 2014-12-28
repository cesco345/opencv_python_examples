#!/usr/bin/env python

import numpy as np
import cv2
import video

help_message = '''
USAGE: my_warp_flow.py [<video_source>]

'''

def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res

if __name__ == '__main__':
    import sys
    print help_message
    try: fn = sys.argv[1]
    except: fn = 0

    cam = video.create_capture(fn)
    ret, prev = cam.read()
    prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    cur_glitch = prev.copy()

    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prevgray, gray, 0.5, 3, 15, 3, 5, 1.2, 0)
        prevgray = gray
        cur_glitch = warp_flow(cur_glitch, flow)
        cv2.imshow('glitch', cur_glitch)

        ch = 0xFF & cv2.waitKey(5)
        if ch == 27:
            break
        #cur_glitch = img.copy()
    cv2.destroyAllWindows()

