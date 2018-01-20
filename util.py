import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
import sys
from time import time

# Check whether the script is being called from the command line properly
def check_usage(script_name):
  if len(sys.argv) != 2:
    print "Usage: python %s.py <filename>" % script_name
    exit(1)

# Code in this method modified from:
# http://docs.opencv.org/trunk/doc/py_tutorials/py_feature2d/py_features_harris/py_features_harris.html
def read_image(filename):
  # filename = sys.argv[1]
  if not os.path.isfile(filename):
    print "No such file:", filename
    exit(1)

  img = cv2.imread(filename)
  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  return img, gray

def wait_until_done():
  if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()

def vec_to_img(vec,img):
  H = img.shape[0]
  W = img.shape[1]
  zeros = np.zeros([H,W]) 
  N = vec.shape[0]
  for i in xrange(N):
    x = vec[i][1]
    y = vec[i][0]
    zeros[x,y] = 1
  return zeros

def img_to_vec(img):
  H,W = img.shape
  vec = list()
  for y in xrange(H):
    for x in xrange(W):
      if img[y,x] != 0:
        vec.append([x,y])
  vec = np.array(vec)
  return vec



def distance(p1, p2):
  y, x = p1
  v, u = p2
  return ((y - v) ** 2 + (x - u) ** 2) ** 0.5
