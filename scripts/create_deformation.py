#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pyminc.volumes.factory import *
#from numpy import *
import numpy
from scipy import ndimage
from optparse import OptionParser
import tempfile
import os
import functools
import re
import time
import sys
import math
#import pyximport; pyximport.install()
import cython_code

# make sure input file is in minc2 format
# pass on input file if it is, otherwise 
# create a minc2 version
def impose_minc2_ness(minc_file, temp_dir):
  mincversion = os.popen("mincinfo -minc_version %s" % (minc_file)).readlines()[-1]
  if 'HDF5' in mincversion:
    return minc_file
  # convert the file to the minc2 format
  else:
    print "\nThe input file is not in MINC2 format, converting now..."
    filename = os.path.basename(minc_file)
    basename = filename.replace('.mnc', '')
    minc_2_output = "%s/%s_minc2.mnc" % (temp_dir, basename)
    command = "mincconvert -clobber -2 %s %s" % \
               (minc_file, minc_2_output)
    if g_verbose:
      print "\n%s" % command
    os.system(command)
    if not os.access(minc_2_output,0):
      print "\nERROR: could not create minc2 volume %s!" % minc_2_output
      exit(1)
    else:
      print "done!"
    return minc_2_output

# make sure input file has the z,y,x dimension order
def impose_zyx_ness(minc_file, temp_dir):
  dimorder = os.popen("mincinfo -attvalue image:dimorder  %s" % (minc_file)).readlines()[-1]
  dimorder_array = dimorder.split(',')
  if(not(dimorder_array[0] == "zspace" and dimorder_array[1] == "yspace" and dimorder_array[2] == "xspace")):
    print "\n The dimension order of %s is not in standard (zspace,yspace,xspace) order, converting now..." % minc_file
    filename = os.path.basename(minc_file)
    basename = filename.replace('.mnc', '')
    zyx_output = "%s/%s_zyx.mnc" % (temp_dir, basename)
    command = "mincreshape -double -dimorder zspace,yspace,xspace -clobber -2 %s %s" % \
               (minc_file, zyx_output)
    if g_verbose:
      print "\n%s" % command
    os.system(command)
    if not os.access(zyx_output,0):
      print "\nERROR: could not create volume with the correct dimension order %s!" % zyx_output
      exit(1)
    else:
      print "done!"
    return zyx_output
  else:
    return minc_file


# call create_initial_deformation_grid to create a grid with all zero vectors
def create_initial_deformation(minc_file, temp_dir):
  filename = os.path.basename(minc_file)
  basename = filename.replace('.mnc', '')
  grid_output = "%s/%s_grid.mnc" % (temp_dir, basename)
 
  command = "create_initial_deformation_grid -clobber %s %s" \
            % (minc_file, grid_output)
  if g_verbose:
    print("\n%s" % command)
  os.system(command)
  return grid_output


def get_determinant_from_grid(grid, output):
  tempfile = output.replace('.mnc', '_temp.mnc')
  command = "mincblob -clobber -determinant %s %s" % (grid, tempfile)
  if g_verbose:
    print("\n%s" % command)
  os.system(command)
  command = "mincmath -quiet -clobber -const 1 -add %s %s" % (tempfile, output)
  if g_verbose:
    print("\n%s" % command)
  os.system(command)
  return output
  


###############################################################################
#################################### MAIN #####################################
###############################################################################

if __name__ == "__main__":
  usage = "usage: %prog [options] input.mnc output_deformation.xfm"
  parser = OptionParser(usage)
  g_program = sys.argv[0]


  parser.add_option("-t", "--tolerance", dest="tolerance",
                    help="Specify the amount of error that is allowed between the specified determinant and the final determinant (per voxel) (default = 0.02)",
                    type="float", default=0.02)
  parser.add_option("-i", "--iterations", dest="iterations",
                    help="Specify the maximum number of iterations to update the deformations field (-1 means until convergence) (default = -1)",
                    type="int", default=-1)
  parser.add_option("-n", "--neighbors", dest="neighbors",
                    help="Specify the number of neighbors to use in the determinant calculation (possibilities: 6, 14) (default = 6)",
                    type="int", default=6)
  parser.add_option("-m", "--mask", dest="maptolerance",
                    help="Specify a tolerance map file (.mnc) indicating voxels that have a different amount of error allowed (e.g., CSF, background).",
                    type="string", default=-1)
  #parser.add_option("-s", "--simulatedannealing",
                  #action="store_true", dest="simulatedannealing", default=False,
                  #help="Use simulated annealing to create the deformation field")
  parser.add_option("-v", "--verbose",
                  action="store_true", dest="verbose", default=False,
                  help="Maximize output messages")
  parser.add_option("-k", "--keeptemp",
                  action="store_true", dest="keeptemp", default=False,
                  help="Keep temporary files")
                  



  (options, args) = parser.parse_args()
  g_verbose = options.verbose

  ##########################
  ###### handle input ######
  ##########################

  if len(args) != 2:
    parser.error("incorrect number of arguments, specify an input and output file")
  
  g_source = args[0]
  g_output_xfm = args[1]
  
  # create a directory for temporary files
  g_pwd = os.getcwd();
  counter = 0
  g_tmpdir = "%s/tmp_deformation_%s" % (g_pwd,counter)
  while os.access(g_tmpdir, 0):
    counter = counter + 1
    g_tmpdir = "%s/tmp_deformation_%s" % (g_pwd,counter)
  os.mkdir(g_tmpdir)
  
  # check whether the input file is minc2, if not create a 
  # minc2 version of it
  g_minc2_input = impose_minc2_ness(g_source, g_tmpdir)
    
  # create the inital deformation grid (with all zero vectors)
  g_initial_deformation_grid = create_initial_deformation(g_minc2_input, g_tmpdir)

  ##########################
  ###### loading data ######
  ##########################

  # Load data in memory
  # 1) the input determinant file
  g_target_determinant = volumeFromFile(g_minc2_input, dtype="double")
  g_target_determinant.data
  # 3) a copy of the grid file which will hold the evolving grid
  g_initial_grid = volumeFromFile(g_initial_deformation_grid, dtype="double")
  g_initial_grid.data
  g_filename = os.path.basename(g_initial_deformation_grid)
  g_basename = g_filename.replace('.mnc', '')
  g_evolving_grid_name = "%s/%s_evolving.mnc" % (g_tmpdir, g_basename)
  g_evolving_grid = volumeFromInstance(g_initial_grid,
                                          g_evolving_grid_name,
                                          dtype="double",
                                          data=True,
                                          volumeType="ushort")
  g_evolving_grid.data
  g_evolving_grid.data[:,:,:,:] = 0
  # 3) a copy of the input determinant file which will hold the evolving
  #    determinant file
  g_filename = os.path.basename(g_evolving_grid.filename)
  g_basename = g_filename.replace('.mnc', '')
  g_evolving_determinant_name = "%s/%s_determinant.mnc" % (g_tmpdir, g_basename)
  g_evolving_determinant = volumeFromInstance(g_target_determinant,
                                          g_evolving_determinant_name,
                                          dtype="double",
                                          data=True,
                                          volumeType="ushort")
  g_evolving_determinant.data
  g_evolving_determinant.data[:,:,:] = 0
  
  
  g_xstep = float(g_evolving_determinant.separations[2])
  g_ystep = float(g_evolving_determinant.separations[1])
  g_zstep = float(g_evolving_determinant.separations[0])
  if(options.neighbors == 6):
    cython_code.calculate_determinant_from_grid(g_evolving_grid.data, g_evolving_determinant.data, g_xstep, g_ystep, g_zstep)
  elif(options.neighbors == 14):
    cython_code.calculate_determinant_from_grid_14_neighbors(g_evolving_grid.data, g_evolving_determinant.data, g_xstep, g_ystep, g_zstep)
  else:
    print "Can't calculate the determinant given the number of neighbors: %s" % options.neighbors
    exit(1)
  
  # 4) tolerance map data:
  # initialize data
  
  g_tolerance_map = volumeFromInstance(g_target_determinant,
                                       "/tmp/temp_map_tolerance.mnc",
                                       dtype="double",
                                       data=True,
                                       volumeType="ushort")
  g_tolerance_map.data
  g_tolerance_map.data[:,:,:] = 0
  g_tolerance_map_data = g_tolerance_map.data
  if(options.maptolerance != -1):
    # make sure that the tolerance map has the right dimension order:
    g_maptolerance_right_dimorder = impose_zyx_ness(options.maptolerance, g_tmpdir)    
    g_tolerance_map = volumeFromFile(g_maptolerance_right_dimorder, dtype="double")
    g_tolerance_map_data = g_tolerance_map.data
    g_tol_in_map = g_tolerance_map_data.sum()
    g_tol_neccessary = abs(g_target_determinant.data.sum() - 
                           (g_target_determinant.data.shape[0] *
                            g_target_determinant.data.shape[1] *
                            g_target_determinant.data.shape[2] ))
    g_factor = g_tol_neccessary / g_tol_in_map
    
    #g_tolerance_map_data *= (g_factor + 0.75)
    
    #
    # Just for now: set the tolerance values as requirements and give
    # them a little more tolerance than other voxels
    #
    #for tolz in range(0,g_tolerance_map_data.shape[0]):
      #for toly in range (0,g_tolerance_map_data.shape[1]):
        #for tolx in range (0,g_tolerance_map_data.shape[2]):
          #if(g_tolerance_map_data[tolz,toly,tolx] != 0):
            #g_target_determinant.data[tolz,toly,tolx] = g_tolerance_map_data[tolz,toly,tolx]/2 + 1
            #g_tolerance_map_data[tolz,toly,tolx] = g_tolerance_map_data[tolz,toly,tolx]/2 + 0.5
    
  
  
  ###########################
  ###########################
  #### Core Calculations ####
  ###########################
  ###########################
  
  g_evolving_grid_data = g_evolving_grid.data
  g_field_difference = cython_code.get_initial_difference(g_target_determinant.data,
                                                          g_evolving_determinant.data,
                                                          g_tolerance_map_data,
                                                          options.tolerance,
                                                          g_xstep,
                                                          g_ystep,
                                                          g_zstep) + 1
  g_previous_field_diff = g_field_difference
  g_max_field_difference = g_field_difference
  #g_derivative_factor =  math.sin(math.pi/2 * g_previous_field_diff/g_max_field_difference)
  g_derivative_factor = g_previous_field_diff/g_max_field_difference
  g_iterations = options.iterations
  if g_iterations == -1:
    g_iterations = 999999
  #g_extra_derivative_weight = g_max_field_difference * 3  / g_iterations
  #### update grid:
  for i in range(1,g_iterations):
    if (g_field_difference > 0):
      g_derivative_factor = ((0.25*i/g_iterations) / (math.exp(0.25*i/g_iterations)-1 ))
      g_field_difference = cython_code.update_grid(g_target_determinant.data,
                                                   g_evolving_determinant.data,
                                                   g_evolving_grid_data,
                                                   g_tolerance_map_data,
                                                   options.tolerance,
                                                   g_xstep,
                                                   g_ystep,
                                                   g_zstep,
                                                   g_derivative_factor)
      g_current_derivative = abs(g_previous_field_diff - g_field_difference)
      if(g_field_difference > g_max_field_difference):
        g_max_field_difference = g_field_difference
        #g_extra_derivative_weight = g_max_field_difference * 3  / g_iterations
      if(i % 10 == 0):
        print "Iteration %5.0f determinant differnce: %3.5f derivative factor used: %3.5f" % \
          (i, g_field_difference, g_derivative_factor)
        # write out the current deformation grid
        counter = 0
        g_evolving_grid_intermediate_name = "%s/intermediate_grid_%07d.mnc" % (g_pwd,counter)
        while os.access(g_evolving_grid_intermediate_name, 0):
          counter = counter + 1
          g_evolving_grid_intermediate_name = "%s/intermediate_grid_%07d.mnc" % (g_pwd,counter)
        g_evolving_grid_intermediate = volumeFromInstance(g_evolving_grid,
                                          g_evolving_grid_intermediate_name,
                                          dtype="double",
                                          data=True,
                                          volumeType="ushort")
        g_evolving_grid_intermediate.data
        g_evolving_grid_intermediate.writeFile()
        g_evolving_grid_intermediate.closeVolume()
      if(options.neighbors == 6):
        cython_code.calculate_determinant_from_grid(g_evolving_grid_data, g_evolving_determinant.data, g_xstep, g_ystep, g_zstep)
      elif(options.neighbors == 14):
        cython_code.calculate_determinant_from_grid_14_neighbors(g_evolving_grid_data, g_evolving_determinant.data, g_xstep, g_ystep, g_zstep)
      g_previous_field_diff = g_field_difference
      #g_derivative_factor =  math.sin(math.pi/2 * g_previous_field_diff/g_max_field_difference)
      #g_derivative_factor = g_previous_field_diff/(g_max_field_difference + (g_extra_derivative_weight * i))
      if(i < 7500):
        g_derivative_factor = 0.9 #g_previous_field_diff/(g_max_field_difference + (g_extra_derivative_weight * i))
      elif(i < 15000):
        g_derivative_factor = 0.5
      elif(i < 22500):
        g_derivative_factor = 0.2
      else:
        g_derivative_factor = 0.01
      
      #g_old_grid_data = g_evolving_grid_data
      #g_evolving_grid_data = g_temp_grid_data

  
  g_evolving_grid.writeFile()
  g_evolving_grid.closeVolume()
  g_evolving_determinant.writeFile()
  g_evolving_determinant.closeVolume()
  
  #### create the transform
  ## first copy the grid
  g_xfm_grid = g_output_xfm
  g_xfm_grid = g_xfm_grid.replace('.xfm', '_grid.mnc')
  os.system("cp -f %s %s" % (g_evolving_grid.filename, g_xfm_grid))
  ## create the xfm file
  g_xfm = open(g_output_xfm, 'w')
  g_xfm.write("MNI Transform File\n")
  g_current_date_time = time.strftime("%Y/%m/%d %H:%M:%S", time.localtime())
  g_xfm.write("%% File automatically created by %s\n" % g_program)
  g_xfm.write("%% %s \n" % g_current_date_time)
  g_xfm.write("%% Input file: %s \n" % g_source)
  g_xfm.write("%% Tolerance: %s \n" % options.tolerance)
  g_xfm.write("%% Iterations: %s \n" % g_iterations)
  if(options.maptolerance != -1):
    g_xfm.write("%% Tolerance map: %s \n" % options.maptolerance)
  else:
    g_xfm.write("%% No tolerance map was used\n")
  g_xfm.write("%% Number of neighbors to calculate determinant: %s \n" % options.neighbors)
  g_xfm.write("Transform_Type = Grid_Transform;\n")
  ## get only the basename of the file...
  g_base_of_g_xfm_grid = os.path.basename(g_xfm_grid)
  g_xfm.write("Displacement_Volume = %s;\n" % g_base_of_g_xfm_grid)
  
  #if not options.keeptemp:
    #os.system("rm -fr %s" % g_tmpdir)
    
  

  print "\ndone!\n\n"
