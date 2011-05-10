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
    os.system(command)
    if not os.access(minc_2_output,0):
      print "\nERROR: could not create minc2 volume %s!" % minc_2_output
      exit(1)
    else:
      print "done!"
    return minc_2_output


# This function makes a copy of the input minc volume, and creates
# the desired determinant field out of the copy. It returns the
# filename of the determinant file.
#
# source             - a minc volume read through pyminc
# center_deformation - a comma seperated string specifying x,y,z
# tmpdir             - string of the temp directory
# determinant_value  - value that the deformation field should have in the ROI
# sphere_radius      - radius of the deformation area
#
def create_initial_determinant(source, center_deformation, tmpdir, outputfilename,
                               determinant_value=None, sphere_radius=None):
  determinant_volume = volumeFromInstance(source, outputfilename)
  determinant_volume.data[:,:,:] = 1
  
  # write out a 3D "sphere"
  # first we loop over one of the axis (z)
  for z in range(-sphere_radius, sphere_radius+1):
    # "draw" a 2D circle each time with the largest
    # circle when y=0; middle of the sphere
    circle_radius = sphere_radius - abs(z)
    for y in range(-circle_radius, circle_radius+1):
      # other axis of the circle
      yaxisradius  = circle_radius - abs(y)
      for x in range(-yaxisradius, yaxisradius+1):
        #print "x: %s y: %s z: %s" % (x,y,z)
        determinant_volume.data[center_deformation[2] - z, 
                                center_deformation[1] - y,
                                center_deformation[0] - x] = determinant_value
  
  determinant_volume.writeFile()
  determinant_volume.closeVolume()
  return outputfilename


###############################################################################
#################################### MAIN #####################################
###############################################################################

if __name__ == "__main__":
  usage = "\n\nThis program creates a determinant file for the given input file.\n\
The deformation area has the shape of a sphere, and the center, \n\
radius and determinant value of the area can be specified.\n\
\n\
usage: %prog [options] input.mnc output_determinant.mnc"
  parser = OptionParser(usage)
  g_program = sys.argv[0]

  parser.add_option("-c", "--center", dest="center",
                    help="Specify the center of the deformation area x,y,z",
                    type="string")
  parser.add_option("-d", "--determinant-value", dest="determinant",
                    help="Specify the determinant value of the deformation area",
                    type="float")
  parser.add_option("-r", "--radius", dest="radius",
                    help="Specify the radius of the deformation area in voxels",
                    type="int")
  parser.add_option("-k", "--keeptemp",
                  action="store_true", dest="keeptemp", default=False,
                  help="Keep temporary files")

  (options, args) = parser.parse_args()

  ##########################
  ###### handle input ######
  ##########################

  if len(args) != 2:
    parser.error("incorrect number of arguments, specify an input and output file")

  g_source = args[0]
  g_output = args[1]
  
  # create a directory for temporary files
  g_pwd = os.getcwd();
  g_tmpdir = "%s/tmp_determinant" % g_pwd
  if not os.access(g_tmpdir, 0):
    os.mkdir(g_tmpdir)
  
  # check whether the input file is minc2, if not create a 
  # minc2 version of it
  g_minc2_input = impose_minc2_ness(g_source, g_tmpdir)
  print "\nFile: %s" % g_minc2_input
  g_input_volume = volumeFromFile(g_minc2_input)
  g_xindex = 0 if('xspace' in g_input_volume.dimnames[0]) else \
    (1 if ('xspace' in g_input_volume.dimnames[1]) else \
    (2 if ('xspace' in g_input_volume.dimnames[2]) else -1))
  g_yindex = 0 if('yspace' in g_input_volume.dimnames[0]) else \
    (1 if ('yspace' in g_input_volume.dimnames[1]) else \
    (2 if ('yspace' in g_input_volume.dimnames[2]) else -1))
  g_zindex = 0 if('zspace' in g_input_volume.dimnames[0]) else \
    (1 if ('zspace' in g_input_volume.dimnames[1]) else \
    (2 if ('zspace' in g_input_volume.dimnames[2]) else -1))
    
  if(g_xindex == -1 or g_yindex == -1 or g_zindex == -1):
    sys.exit("\nCould not determine each of the 3 dimensions (xspace, yspace, zspace).")


  g_min_dim_length = min(g_input_volume.sizes[0],
                         g_input_volume.sizes[1],
                         g_input_volume.sizes[2])
  g_middle_dim_length = int(g_min_dim_length/2)
  if options.center == None:
    # determine a center for the deformation
    options.center = "%s,%s,%s" % (g_middle_dim_length, g_middle_dim_length, g_middle_dim_length)
    print "Center for deformation not give, set center is: %s" % options.center

  if options.determinant == None:
    print "The determinant value for the deformation area is not give, default = 0.6."
    options.determinant = 0.6

  if options.radius == None:
    options.radius = int(g_middle_dim_length / 4)
    print "The radius of the deformation area is not given, set to %s." % options.radius


  ##########################
  ### feasibility check ####
  ##########################

  # check whether the deformation area fits in the volume
  g_center_array = options.center.split(',')
  g_center = []
  g_center.append(int(g_center_array[0]))
  g_center.append(int(g_center_array[1]))
  g_center.append(int(g_center_array[2]))
  if((g_center[0] - options.radius) < 0 or
     (g_center[1] - options.radius) < 0 or
     (g_center[2] - options.radius) < 0 or
      g_input_volume.sizes[g_xindex] < (g_center[0] + options.radius) or
      g_input_volume.sizes[g_yindex] < (g_center[1] + options.radius) or
      g_input_volume.sizes[g_zindex] < (g_center[2] + options.radius)) :
    sys.exit("\nThe deformation area does not fit in the volume.")
  
  ##########################
  ### Create determinant  ##
  ##########################
  
  g_eat_output_text = g_input_volume.data
  g_determinant_file = create_initial_determinant(g_input_volume,
                                                  g_center, 
                                                  g_tmpdir, 
                                                  g_output,
                                                  options.determinant, 
                                                  options.radius)
  
  if not options.keeptemp:
    os.system("rm -fr %s" % g_tmpdir)

  print "\ndone!\n\n"
