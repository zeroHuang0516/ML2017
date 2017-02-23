#!/usr/bin/env python
from PIL import Image
import numpy as np
import sys
im1 = Image.open(sys.argv[1])
im2 = Image.open(sys.argv[2])
im1 = im1.convert("RGBA")
im2 = im2.convert("RGBA")
width, height = im1.size
pixdata1 = im1.load()
pixdata2 = im2.load()
for i in range(width):
  for j in range(height):
      if(pixdata1[i,j]!=pixdata2[i,j] ):
        pixdata1[i, j] = pixdata2[i,j]
      else:
        pixdata1[i, j] = (0, 0, 0, 0)
im1.save("ans_two.png","PNG")
