#!/usr/bin/env python

from pyminc.volumes.factory import *
from numpy import *
from scipy import ndimage
from optparse import OptionParser
import tempfile
import os
import functools
import re
import time
import sys

def calculate_determinant_from_grid(grid, determinant):
  
  # 3x3 array to hold the Jacobian matrix
  jac = zeros((3,3))
  
  # exclude the outer borders:
  zmax = determinant.sizes[0]-1
  ymax = determinant.sizes[1]-1
  xmax = determinant.sizes[2]-1
  zsteps = determinant.separations[2]
  ysteps = determinant.separations[1]
  xsteps = determinant.separations[0]
  
  for z in range(1, determinant.sizes[0]-1):
    for y in range(1, determinant.sizes[1]-1):
      for x in range(1, determinant.sizes[2]-1):
        jac[0,0] = 1 + ((grid.data[0,z,y,x+1] - 
                         grid.data[0,z,y,x-1]) / (determinant.separations[2] * 2))
        jac[0,1] = (grid.data[0,z,y+1,x] - 
                    grid.data[0,z,y-1,x]) / (determinant.separations[1] * 2)
        jac[0,2] = (grid.data[0,z+1,y,x] - 
                    grid.data[0,z-1,y,x]) / (determinant.separations[0] * 2)
        jac[1,0] = (grid.data[1,z,y,x+1] - 
                    grid.data[1,z,y,x-1]) / (determinant.separations[2] * 2)
        jac[1,1] = 1 + ((grid.data[1,z,y+1,x] - 
                         grid.data[1,z,y-1,x]) / (determinant.separations[1] * 2))
        jac[1,2] = (grid.data[1,z+1,y,x] - 
                    grid.data[1,z-1,y,x]) / (determinant.separations[0] * 2)
        jac[2,0] = (grid.data[2,z,y,x+1] - 
                    grid.data[2,z,y,x-1]) / (determinant.separations[2] * 2)
        jac[2,1] = (grid.data[2,z,y+1,x] - 
                    grid.data[2,z,y-1,x]) / (determinant.separations[1] * 2)
        jac[2,2] = 1 + ((grid.data[2,z+1,y,x] - 
                         grid.data[2,z-1,y,x]) / (determinant.separations[0] * 2))
        
        determinant.data[z,y,x] = \
          (jac[0,0] * ((jac[1,1] * jac[2,2]) - (jac[1,2] * jac[2,1])) -
           jac[0,1] * ((jac[1,0] * jac[2,2]) - (jac[1,2] * jac[2,0])) +
           jac[0,2] * ((jac[1,0] * jac[2,1]) - (jac[1,1] * jac[2,0])))
  
  
  #Deal with the borders
  #Default value for the determinant is 1 (meaning no change)