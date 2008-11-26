#!/usr/bin/env python

from pyminc.volumes.factory import *
from numpy import *
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
    return minc_2_output

# a call to create_determinant_file, a program that creates 
# the desired jacobian determinant file (unblurred)
def create_initial_determinant(source, center_deformation, tmpdir,
                               determinant_value=None, sphere_radius=None):
  filename = os.path.basename(source)
  basename = filename.replace('.mnc','')
  outputname = "%s/%s_sharp_det.mnc" % (tmpdir, basename)
  command = "create_determinant_file -clobber -center %s -radius %d -determinant %s %s %s" \
            % (center_deformation, sphere_radius, determinant_value, source, \
               outputname)
  if g_verbose:
    print("\n%s" % command)
  os.system(command)
  return outputname

# pad a minc file with 10 layers of voxels all around the volume
def pad_minc_file(minc_file):
  xlength = os.popen("mincinfo -dimlength xspace %s" % minc_file).readlines()[-1]
  ylength = os.popen("mincinfo -dimlength yspace %s" % minc_file).readlines()[-1]
  zlength = os.popen("mincinfo -dimlength zspace %s" % minc_file).readlines()[-1]
  xlength = 20 + int(xlength)
  ylength = 20 + int(ylength)
  zlength = 20 + int(zlength)
  outputname = minc_file.replace('.mnc', '_padded.mnc')
  # minc dimension order: zspace yspace xspace
  command = "mincreshape -clobber -fill -fillvalue 1 -start -10,-10,-10 -count %s,%s,%s %s %s" \
            % (zlength, ylength, xlength, minc_file, outputname)
  if g_verbose:
    print("\n%s" % command)
  os.system(command)
  return outputname

# blur the input minc file with the given kernel
def blur_minc_file(minc_file, blur_kernel):
  basename = minc_file.replace('.mnc', '')
  blurbase = "%s_fwhm_%s" % (basename, blur_kernel)
  bluroutput = "%s_blur.mnc" % (blurbase)
  command = "mincblur -clobber -no_apodize -fwhm %s %s %s" \
            % (blur_kernel, minc_file, blurbase)
  if g_verbose:
   print("\n%s" % command)
  os.system(command)
  if not os.access(bluroutput,0):
    print "\nERROR: could not create blurred volume %s!" % bluroutput
    exit(1)
  return bluroutput

# unpad a minc file. Remove 10 layers of voxels all around the volume
def unpad_minc_file(minc_file):
  xlength = os.popen("mincinfo -dimlength xspace %s" % minc_file).readlines()[-1]
  ylength = os.popen("mincinfo -dimlength yspace %s" % minc_file).readlines()[-1]
  zlength = os.popen("mincinfo -dimlength zspace %s" % minc_file).readlines()[-1]
  xlength = int(xlength) - 20
  ylength = int(ylength) - 20
  zlength = int(zlength) - 20
  outputname = minc_file.replace('.mnc', '_unpadded.mnc')
  tempoutput = outputname.replace('.mnc', '_temp.mnc')
  # minc dimension order: zspace yspace xspace
  command = "mincreshape -clobber -fill -fillvalue 1 -start 10,10,10 -count %s,%s,%s %s %s" \
            % (zlength, ylength, xlength, minc_file, tempoutput)
  if g_verbose:
    print("\n%s" % command)
  os.system(command)
  # set the correct image ranges
  imagemin = float(os.popen("mincstats -quiet -min %s" % tempoutput).readlines()[-1])
  imagemax = float(os.popen("mincstats -quiet -max %s" % tempoutput).readlines()[-1])
  command = "mincreshape -clobber -image_range %s %s %s %s" \
            % (imagemin, imagemax, tempoutput, outputname)
  if g_verbose:
    print("\n%s" % command)
  os.system(command)
  # remove tempfile...
  os.system("rm -f %s" % tempoutput)
  return outputname

# call create_initial_deformation_grid to create a grid with all zero vectors
def create_initial_deformation(minc_file):
  outputname = minc_file.replace('.mnc', '_grid.mnc')
  command = "create_initial_deformation_grid -clobber %s %s" \
            % (minc_file, outputname)
  if g_verbose:
    print("\n%s" % command)
  os.system(command)
  return outputname


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


if __name__ == "__main__":
  usage = "usage: %prog [options] input.mnc output_deformation.xfm"
  parser = OptionParser(usage)
  g_program = sys.argv[0]

  parser.add_option("-b", "--blurkernel", dest="blurkernel",
                    help="Specify the blurring kernel for the determinant file",
                    type="float")
  parser.add_option("-c", "--center", dest="center",
                    help="Specify the center of the deformation area x,y,z",
                    type="string")
  parser.add_option("-d", "--determinant-value", dest="determinant",
                    help="Specify the determinant value of the deformation area",
                    type="float")
  parser.add_option("-r", "--radius", dest="radius",
                    help="Specify the radius of the deformation area",
                    type="float")
  parser.add_option("-i", "--iterations", dest="iterations",
                    help="Specify the maximum number of iterations to update the deformations field (-1 means until convergence) (default = -1)",
                    type="float", default=-1)
  parser.add_option("-v", "--verbose",
                  action="store_true", dest="verbose", default=False,
                  help="Maximize output messages")
  parser.add_option("-k", "--keeptemp",
                  action="store_true", dest="keeptemp", default=False,
                  help="Keep temporary files")



  (options, args) = parser.parse_args()
  g_verbose = options.verbose

  if len(args) != 2:
    parser.error("incorrect number of arguments")

  if options.blurkernel == None:
    parser.error("blurring kernel for deformation not given")

  if options.center == None:
    parser.error("center for deformation not given")

  if options.determinant == None:
    parser.error("determinant value not given")

  if options.radius == None:
    parser.error("radius of sphere not given")

  g_source = args[0]
  g_output_xfm = args[1]

  # create a directory for temporary files
  g_pwd = os.getcwd();
  g_tmpdir = "%s/tmp_deformation" % g_pwd
  if not os.access(g_tmpdir, 0):
    os.mkdir(g_tmpdir)

  # create the desired jacobian determinant file
  g_determinant_file = create_initial_determinant(g_source,
                                                  options.center, 
                                                  g_tmpdir, 
                                                  options.determinant, 
                                                  options.radius)


  # if blurring is requested, pad the determinant file before blurring
  # to prevent artifacts around the outside of the volume and remove
  # the padding from the file when done blurring
  if options.blurkernel != 0:
    # pad file
    g_padded_determinant = pad_minc_file(g_determinant_file)
    # blur the determinant file
    g_blurred_padded_determinant = blur_minc_file(g_padded_determinant,
                                                  options.blurkernel)
    # unpad the blurred determinant file
    g_blurred_determinant = unpad_minc_file(g_blurred_padded_determinant)
    g_determinant_file = g_blurred_determinant
  
  ## Following code was meant temporary to allow for a specified determinant file
  #g_give_det = args[2]
  #g_filename = os.path.basename(g_give_det)
  #g_outputname = "%s/%s" % (g_tmpdir, g_filename)
  #os.system("cp %s %s" % (g_give_det,g_outputname) )
  #g_determinant_file = g_outputname
  

  # create the inital deformation grid (with all zero vectors)
  g_initial_deformation_grid = create_initial_deformation(g_determinant_file)

  # get the jacobian determinant of the grid
  g_initial_determinant = g_initial_deformation_grid.replace('.mnc', '_det.mnc')
  g_initial_determinant = get_determinant_from_grid(g_initial_deformation_grid,
                                                    g_initial_determinant)
  #initially, the image range of the determinant file are incorrect:
  g_temp_correction = g_initial_determinant.replace('.mnc', '_correct_range.mnc')
  os.system("mv %s %s" % (g_initial_determinant, g_temp_correction))
  os.system("mincreshape -clobber   -image_range -100 100 %s %s" % \
            (g_temp_correction, g_initial_determinant))


  g_current_grid = g_initial_deformation_grid
  g_current_grid = g_current_grid.replace('.mnc', '_current.mnc')
  g_old_grid = g_current_grid.replace('_current.mnc', '_old.mnc')
  g_temp_grid = g_current_grid.replace('_current.mnc', '_temp.mnc')
  os.system("cp -f %s %s" % (g_initial_deformation_grid, g_current_grid))
  g_current_det = g_initial_determinant 
  g_field_difference = 1
  g_iterations = options.iterations
  if g_iterations == -1:
    g_iterations = 99999
  
  
  ### update grid:
  for i in range(1,g_iterations):
    if (g_field_difference > 0):
      command = "update_deformation_grid -clobber %s %s %s %s" \
              % (g_determinant_file, g_current_det, g_current_grid, g_temp_grid)
      if g_verbose:
        print("\n%s" % command)
      g_update_output = os.popen(command).readlines()[-1]
      g_field_difference = array(g_update_output.strip().split(" "))[-1].astype("float32")
      if g_verbose:
        print "Iteration %s: difference between determinant files: %s" % (i, g_field_difference)
      else:
        if(i % 10 == 0):
          print "Iteration %s: difference between determinant files: %s" % (i, g_field_difference)
      g_current_det = get_determinant_from_grid(g_temp_grid,
                                                      g_current_det)
      os.system("mv -f %s %s" % (g_current_grid, g_old_grid))
      os.system("mv -f %s %s" % (g_temp_grid, g_current_grid))
  
  
  ### create the transform
  # first copy the grid
  g_xfm_grid = g_output_xfm
  g_xfm_grid = g_xfm_grid.replace('.xfm', '_grid.mnc')
  os.system("cp -f %s %s" % (g_current_grid, g_xfm_grid))
  # create the xfm file
  g_xfm = open(g_output_xfm, 'w')
  g_xfm.write("MNI Transform File\n")
  g_current_date_time = time.strftime("%Y/%m/%d %H:%M:%S", time.localtime())
  g_xfm.write("%% File automatically created by %s\n" % g_program)
  g_xfm.write("%% %s \n" % g_current_date_time)
  g_xfm.write("Transform_Type = Grid_Transform;\n")
  g_xfm.write("Displacement_Volume = %s;\n" % g_xfm_grid)
  
  if not options.keeptemp:
    os.system("rm -fr %s" % g_tmpdir)

  print "\ndone!\n\n"
