import numpy
cimport numpy
cimport cython
from math import sqrt
from math import exp
import random

DTYPE = numpy.float64
ctypedef numpy.float64_t DTYPE_t

@cython.boundscheck(False)

#
# Input: x,y,z: the position where the current determinant is different 
#               from the target determinant by more than the allowed tolerance
#        diff: allowed tolerance
#        evolvinggriddata: holds the current grid, updated up to this voxel
#
# Output: evolvinggriddata will be updated 
#
# numpy.ndarray[DTYPE_t, ndim=4]
#def update_neighbors(int z, int y, int x, float diff,
                     #float xstep, float ystep, float zstep,
                     #numpy.ndarray[DTYPE_t, ndim=4, mode="c"] evolvinggriddata):
  #cdef double x_vec_length =   xstep / 6 * diff * 0.25
  #cdef double y_vec_length =   ystep / 6 * diff * 0.25
  #cdef double z_vec_length =   zstep / 6 * diff * 0.25
  
  ##update vector dimension neighbors: vec (0,1,2) = (x,y,z)
  ##x-1,y,z   adjust x vector
  #evolvinggriddata[0,z,y,x-1] -= x_vec_length
  ##x+1,y,z   adjust x vector
  #evolvinggriddata[0,z,y,x+1] += x_vec_length
  
  ##x,y-1,z   adjust y vector
  #evolvinggriddata[1,z,y-1,x] -= y_vec_length
  ##x,y+1,z   adjust y vector
  #evolvinggriddata[1,z,y+1,x] += y_vec_length
  
  ##x,y,z-1   adjust z vector
  #evolvinggriddata[2,z-1,y,x] -= z_vec_length
  ##x,y,z+1   adjust z vector
  #evolvinggriddata[2,z+1,y,x] += z_vec_length


#
# The input to this function:
# - the target determinant
# - the current determinant (evolving)
# - the map indicating voxels that are allowed to change semi freely
# - the amount of tolerance a voxel is allowed to have -> difference between target and current
# - xstep, ystep, zstep: resolution of the volume
#
def get_initial_difference(numpy.ndarray[DTYPE_t, ndim=3, mode="c"] targetdetdata,
                numpy.ndarray[DTYPE_t, ndim=3, mode="c"] evolvingdetdata,
                numpy.ndarray[DTYPE_t, ndim=3, mode="c"] tolerancemapdata,
                float tolerance, float xstep, float ystep, float zstep):
  
  # MAIN LOOP
  # length of the 3 dimensions
  cdef int nx = targetdetdata.shape[2]
  cdef int ny = targetdetdata.shape[1]
  cdef int nz = targetdetdata.shape[0]
  cdef int z,y,x
  cdef double diff = 0
  cdef double extra_tolerance = 0
  cdef double total_diff = 0
  for z in range(1, nz-1):
    for y in range(1, ny-1):
      for x in range(1, nx-1):
        # for each voxel calculate its difference with the target
        diff = targetdetdata[z,y,x] - evolvingdetdata[z,y,x]
        if(diff > tolerance or diff < -tolerance):
          # the difference is not within the regular tolerance levels, check extra tolerance
          # see if this voxel is allowed to be different (e.g., ventricle voxel or voxel outside of the brain)
          extra_tolerance = tolerancemapdata[z,y,x]
          if(extra_tolerance == 0):
            # no extra tolerance
            if(diff < 0):
              total_diff += abs(diff + tolerance)
            else: # diff > 0
              total_diff += abs(diff - tolerance)
          else: 
            # there is some tolerance, there is only a problem if it now
            # falls outside of the tolerance area
            if( (1 / (1 + extra_tolerance)) > evolvingdetdata[z,y,x]):
              # we need to get back inside the valid range, the voxel is getting too small
              total_diff += (1 / (1 + extra_tolerance)) - evolvingdetdata[z,y,x]
            if( evolvingdetdata[z,y,x] > 1 + extra_tolerance):
              # we need to get back inside the valid range, the voxel is getting too large
              total_diff += (evolvingdetdata[z,y,x] - 1 - extra_tolerance)
  
  return total_diff


#
# Input: targetdetdata: what the determinant should look like
#        evolvingdetdata: what the current determinant looks like
#        evolvinggriddata: current deformation grid
#       tolerancemapdata: a map that indicates extra error allowed for voxels
#                         for instance in case of CSF voxels
######################        tempgriddata: data that will hold the output grid
#
# Output: tempgriddata will contain the updated deformation grid
#
#
def update_grid(numpy.ndarray[DTYPE_t, ndim=3, mode="c"] targetdetdata,
                numpy.ndarray[DTYPE_t, ndim=3, mode="c"] evolvingdetdata,
                numpy.ndarray[DTYPE_t, ndim=4, mode="c"] evolvinggriddata,
                numpy.ndarray[DTYPE_t, ndim=3, mode="c"] tolerancemapdata,
                float tolerance, float xstep, float ystep, float zstep,
                float derivative):
  
  # MAIN LOOP
  # loop over all voxels (that do not lie on the border
  # of the volume) in the target determinant and 
  # compare it with the current determinant file, update
  # the neighbors when the difference is too large
  cdef int nx = targetdetdata.shape[2]
  cdef int ny = targetdetdata.shape[1]
  cdef int nz = targetdetdata.shape[0]

  cdef int z,y,x
  cdef double diff = 0
  cdef double extra_tolerance = 0
  cdef double total_diff = 0
  cdef double x_vec_length = 0
  cdef double y_vec_length = 0
  cdef double z_vec_length = 0
 
  ##############################################################################
  ### USE A GREEDY ALGORITHM TO SOLVE THE DEFORMATION FIELD ####################
  ##############################################################################
  for z in range(1, nz-1):
    for y in range(1, ny-1):
      for x in range(1, nx-1):
        diff = targetdetdata[z,y,x] - evolvingdetdata[z,y,x]
        extra_tolerance = tolerancemapdata[z,y,x]
        # 
        # if the difference is positive, that means that the current/evolving determinant
        # is smaller than the target determinant. How much smaller is this voxel allowed to be?
        # if the extra tolerance is 1, then that means that the voxels can have a determinant
        # 0.5 smaller than indicated, is it 2, than it can be 0.666 smaller. This comes 
        # down to:
        #
        #  1 - ( 1 / (1 + extra_tolerance))
        #
        # When the difference is negative, the evolving determinant value is larger than the 
        # target. In this case, the amount of change allowed is simply the extra_tolerance
        #
        if (diff > ( tolerance +  ( 1 - ( 1 / (1 + extra_tolerance)))) or diff < ( -tolerance - extra_tolerance ) ):
          # we don't fall within the limits
          # now we are overestimating the difference...
          total_diff += abs(diff) - tolerance
          
          
          # get the needed deformation in equal amounts from 
          # the voxel's neighbors
          x_vec_length =   xstep / 6 * diff * derivative
          y_vec_length =   ystep / 6 * diff * derivative
          z_vec_length =   zstep / 6 * diff * derivative  
        
        
          #update vector dimension neighbors: vec (0,1,2) = (x,y,z)
          #x-1,y,z   adjust x vector
          if(x > 2):
            evolvinggriddata[0,z,y,x-1] -= x_vec_length
          #x+1,y,z   adjust x vector
          if(x < nx-3):
            evolvinggriddata[0,z,y,x+1] += x_vec_length
          #x,y-1,z   adjust y vector
          if(y > 2):
            evolvinggriddata[1,z,y-1,x] -= y_vec_length
          #x,y+1,z   adjust y vector
          if(y < ny-3):
            evolvinggriddata[1,z,y+1,x] += y_vec_length
          #x,y,z-1   adjust z vector
          if(z > 2):
            evolvinggriddata[2,z-1,y,x] -= z_vec_length
          #x,y,z+1   adjust z vector
          if(z < nz-3):
            evolvinggriddata[2,z+1,y,x] += z_vec_length

  return total_diff

################################################################################


#
# Input: griddata: deformation grid
#        determinantdata: data that will hold the determinant 
#        xstep, ystep, zstep: size of the voxels
#
# Output: determinantdata will contain the determinant associated with
#         the griddata calculated using 14 neighbors
#
#
def calculate_determinant_from_grid_14_neighbors(numpy.ndarray[DTYPE_t, ndim=4, mode="c"] griddata, 
                                                 numpy.ndarray[DTYPE_t, ndim=3, mode="c"] determinantdata,
                                                 float xstep, float ystep, float zstep):
  
  # 3x3 array to hold the Jacobian matrix
  cdef numpy.ndarray[numpy.float64_t, ndim=2, mode="c"] jac = numpy.zeros((3,3), "float64")
  
  cdef int nx = determinantdata.shape[2]
  cdef int ny = determinantdata.shape[1]
  cdef int nz = determinantdata.shape[0]
  
  cdef double two_xstep = xstep * 2
  cdef double two_ystep = ystep * 2
  cdef double two_zstep = zstep * 2
  cdef double two_xstep_corner = xstep * 2 * 4
  cdef double two_ystep_corner = ystep * 2 * 4
  cdef double two_zstep_corner = zstep * 2 * 4
  
  # MAIN LOOP
  # calculate the jacobian for all the voxels that are not 
  # on the border of the volume
  cdef int z,y,x
  for z in range(1, nz-1):
    for y in range(1, ny-1):
      for x in range(1, nx-1):
        jac[0,0] = 1 + ((griddata[0,z,y,x+1] - griddata[0,z,y,x-1]) / two_xstep +
                    (griddata[0,z+1,y+1,x+1] - griddata[0,z+1,y+1,x-1]) / two_xstep_corner +
                    (griddata[0,z-1,y+1,x+1] - griddata[0,z-1,y+1,x-1]) / two_xstep_corner +
                    (griddata[0,z+1,y-1,x+1] - griddata[0,z+1,y-1,x-1]) / two_xstep_corner +
                    (griddata[0,z-1,y-1,x+1] - griddata[0,z-1,y-1,x-1]) / two_xstep_corner )/2.0
        jac[0,1] = ((griddata[0,z,y+1,x]     - griddata[0,z,y-1,x]) / two_ystep +
                   (griddata[0,z+1,y+1,x+1] - griddata[0,z+1,y-1,x+1]) / two_ystep_corner +
                   (griddata[0,z-1,y+1,x+1] - griddata[0,z-1,y-1,x+1]) / two_ystep_corner +
                   (griddata[0,z+1,y+1,x-1] - griddata[0,z+1,y-1,x-1]) / two_ystep_corner +
                   (griddata[0,z-1,y+1,x-1] - griddata[0,z-1,y-1,x-1]) / two_ystep_corner )/2.0
        jac[0,2] = ((griddata[0,z+1,y,x]     - griddata[0,z-1,y,x]) / two_zstep +
                   (griddata[0,z+1,y+1,x+1] - griddata[0,z-1,y+1,x+1]) / two_zstep_corner +
                   (griddata[0,z+1,y-1,x+1] - griddata[0,z-1,y-1,x+1]) / two_zstep_corner +
                   (griddata[0,z+1,y+1,x-1] - griddata[0,z-1,y+1,x-1]) / two_zstep_corner +
                   (griddata[0,z+1,y-1,x-1] - griddata[0,z-1,y-1,x-1]) / two_zstep_corner )/2.0
        
        jac[1,0] = ((griddata[1,z,y,x+1]     - griddata[1,z,y,x-1]) / two_xstep +
                   (griddata[1,z+1,y+1,x+1] - griddata[1,z+1,y+1,x-1]) / two_xstep_corner +
                   (griddata[1,z-1,y+1,x+1] - griddata[1,z-1,y+1,x-1]) / two_xstep_corner +
                   (griddata[1,z+1,y-1,x+1] - griddata[1,z+1,y-1,x-1]) / two_xstep_corner +
                   (griddata[1,z-1,y-1,x+1] - griddata[1,z-1,y-1,x-1]) / two_xstep_corner )/2.0
        jac[1,1] = 1 + ((griddata[1,z,y+1,x] - griddata[1,z,y-1,x]) / two_ystep + 
                    (griddata[1,z+1,y+1,x+1] - griddata[1,z+1,y-1,x+1]) / two_ystep_corner +
                    (griddata[1,z-1,y+1,x+1] - griddata[1,z-1,y-1,x+1]) / two_ystep_corner + 
                    (griddata[1,z+1,y+1,x-1] - griddata[1,z+1,y-1,x-1]) / two_ystep_corner + 
                    (griddata[1,z-1,y+1,x-1] - griddata[1,z-1,y-1,x-1]) / two_ystep_corner )/2.0
        jac[1,2] = ((griddata[1,z+1,y,x]     - griddata[1,z-1,y,x]) / two_zstep +
                   (griddata[1,z+1,y+1,x+1] - griddata[1,z-1,y+1,x+1]) / two_zstep_corner +
                   (griddata[1,z+1,y-1,x+1] - griddata[1,z-1,y-1,x+1]) / two_zstep_corner +
                   (griddata[1,z+1,y+1,x-1] - griddata[1,z-1,y+1,x-1]) / two_zstep_corner +
                   (griddata[1,z+1,y-1,x-1] - griddata[1,z-1,y-1,x-1]) / two_zstep_corner )/2.0

        jac[2,0] = ((griddata[2,z,y,x+1]     - griddata[2,z,y,x-1]) / two_xstep +
                   (griddata[2,z+1,y+1,x+1] - griddata[2,z+1,y+1,x-1]) / two_xstep_corner +
                   (griddata[2,z-1,y+1,x+1] - griddata[2,z-1,y+1,x-1]) / two_xstep_corner +
                   (griddata[2,z+1,y-1,x+1] - griddata[2,z+1,y-1,x-1]) / two_xstep_corner +
                   (griddata[2,z-1,y-1,x+1] - griddata[2,z-1,y-1,x-1]) / two_xstep_corner )/2.0
        jac[2,1] = ((griddata[2,z,y+1,x]     - griddata[2,z,y-1,x]) / two_ystep +
                   (griddata[2,z+1,y+1,x+1] - griddata[2,z+1,y-1,x+1]) / two_ystep_corner +
                   (griddata[2,z-1,y+1,x+1] - griddata[2,z-1,y-1,x+1]) / two_ystep_corner +
                   (griddata[2,z+1,y+1,x-1] - griddata[2,z+1,y-1,x-1]) / two_ystep_corner +
                   (griddata[2,z-1,y+1,x-1] - griddata[2,z-1,y-1,x-1]) / two_ystep_corner )/2.0
        jac[2,2] = 1 + ((griddata[2,z+1,y,x] - griddata[2,z-1,y,x]) / two_zstep +
                    (griddata[2,z+1,y+1,x+1] - griddata[2,z-1,y+1,x+1]) / two_zstep_corner +
                    (griddata[2,z+1,y-1,x+1] - griddata[2,z-1,y-1,x+1]) / two_zstep_corner +
                    (griddata[2,z+1,y+1,x-1] - griddata[2,z-1,y+1,x-1]) / two_zstep_corner +
                    (griddata[2,z+1,y-1,x-1] - griddata[2,z-1,y-1,x-1]) / two_zstep_corner )/2.0
        
        determinantdata[z,y,x] = \
          (jac[0,0] * ((jac[1,1] * jac[2,2]) - (jac[1,2] * jac[2,1])) -
           jac[0,1] * ((jac[1,0] * jac[2,2]) - (jac[1,2] * jac[2,0])) +
           jac[0,2] * ((jac[1,0] * jac[2,1]) - (jac[1,1] * jac[2,0])))
  
  
  #Deal with the borders
  #Default value for the determinant is 1 (meaning no change)
  determinantdata[0,:,:] = 1
  determinantdata[:,0,:] = 1
  determinantdata[:,:,0] = 1
  determinantdata[nz-1,:,:] = 1
  determinantdata[:,ny-1,:] = 1
  determinantdata[:,:,nx-1] = 1
  
  return 0






#
# Input: griddata: deformation grid
#        determinantdata: data that will hold the determinant 
#        xstep, ystep, zstep: size of the voxels
#
# Output: determinantdata will contain the determinant associated with
#         the griddata
#
#
def calculate_determinant_from_grid(numpy.ndarray[DTYPE_t, ndim=4, mode="c"] griddata, 
                                    numpy.ndarray[DTYPE_t, ndim=3, mode="c"] determinantdata,
                                    float xstep, float ystep, float zstep):
  
  # 3x3 array to hold the Jacobian matrix
  cdef numpy.ndarray[numpy.float64_t, ndim=2, mode="c"] jac = numpy.zeros((3,3), "float64")
  
  cdef int nx = determinantdata.shape[2]
  cdef int ny = determinantdata.shape[1]
  cdef int nz = determinantdata.shape[0]
  
  cdef double two_xstep = xstep * 2
  cdef double two_ystep = ystep * 2
  cdef double two_zstep = zstep * 2
  
  # MAIN LOOP
  # calculate the jacobian for all the voxels that are not 
  # on the border of the volume
  cdef int z,y,x
  for z in range(1, nz-1):
    for y in range(1, ny-1):
      for x in range(1, nx-1):
        jac[0,0] = 1 + ((griddata[0,z,y,x+1] - 
                         griddata[0,z,y,x-1]) / two_xstep)
        jac[0,1] = (griddata[0,z,y+1,x] - 
                    griddata[0,z,y-1,x]) / two_ystep
        jac[0,2] = (griddata[0,z+1,y,x] - 
                    griddata[0,z-1,y,x]) / two_zstep
        jac[1,0] = (griddata[1,z,y,x+1] - 
                    griddata[1,z,y,x-1]) / two_xstep
        jac[1,1] = 1 + ((griddata[1,z,y+1,x] - 
                         griddata[1,z,y-1,x]) / two_ystep)
        jac[1,2] = (griddata[1,z+1,y,x] - 
                    griddata[1,z-1,y,x]) / two_zstep
        jac[2,0] = (griddata[2,z,y,x+1] - 
                    griddata[2,z,y,x-1]) / two_xstep
        jac[2,1] = (griddata[2,z,y+1,x] - 
                    griddata[2,z,y-1,x]) / two_ystep
        jac[2,2] = 1 + ((griddata[2,z+1,y,x] - 
                         griddata[2,z-1,y,x]) / two_zstep)
        
        determinantdata[z,y,x] = \
          (jac[0,0] * ((jac[1,1] * jac[2,2]) - (jac[1,2] * jac[2,1])) -
           jac[0,1] * ((jac[1,0] * jac[2,2]) - (jac[1,2] * jac[2,0])) +
           jac[0,2] * ((jac[1,0] * jac[2,1]) - (jac[1,1] * jac[2,0])))
  
  
  #Deal with the borders
  #Default value for the determinant is 1 (meaning no change)
  determinantdata[0,:,:] = 1
  determinantdata[:,0,:] = 1
  determinantdata[:,:,0] = 1
  determinantdata[nz-1,:,:] = 1
  determinantdata[:,ny-1,:] = 1
  determinantdata[:,:,nx-1] = 1
  
  return 0
  