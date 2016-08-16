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
    print("\nThe input file is not in MINC2 format, converting now...")
    filename = os.path.basename(minc_file)
    basename = filename.replace('.mnc', '')
    minc_2_output = "%s/%s_minc2.mnc" % (temp_dir, basename)
    command = "mincconvert -clobber -2 %s %s" % \
               (minc_file, minc_2_output)
    if g_verbose:
      print("\n%s" % command)
    os.system(command)
    if not os.access(minc_2_output,0):
      print("\nERROR: could not create minc2 volume %s!" % minc_2_output)
      exit(1)
    else:
      print("done!")
    return minc_2_output


# pad a minc file with 10 layers of voxels all around the volume
def pad_minc_file(minc_file, temp_dir):
  xlength = os.popen("mincinfo -dimlength xspace %s" % minc_file).readlines()[-1]
  ylength = os.popen("mincinfo -dimlength yspace %s" % minc_file).readlines()[-1]
  zlength = os.popen("mincinfo -dimlength zspace %s" % minc_file).readlines()[-1]
  xlength = 20 + int(xlength)
  ylength = 20 + int(ylength)
  zlength = 20 + int(zlength)
  filename = os.path.basename(minc_file)
  basename = filename.replace('.mnc', '')
  padded_output = "%s/%s_padded.mnc" % (temp_dir, basename)
  # minc dimension order: zspace yspace xspace
  command = "mincreshape -clobber -fill -fillvalue 1 -start -10,-10,-10 -count %s,%s,%s %s %s" \
            % (zlength, ylength, xlength, minc_file, padded_output)
  os.system(command)
  return padded_output

# blur the input minc file with the given kernel
def blur_minc_file(minc_file, blur_kernel, temp_dir):
  filename = os.path.basename(minc_file)
  basename = filename.replace('.mnc', '')
  blurbase = "%s/%s_fwhm_%s" % (temp_dir, basename, blur_kernel)
  bluroutput = "%s_blur.mnc" % (blurbase)
  command = "mincblur -clobber -no_apodize -fwhm %s %s %s" \
            % (blur_kernel, minc_file, blurbase)
  os.system(command)
  if not os.access(bluroutput,0):
    print("\nERROR: could not create blurred volume %s!" % bluroutput)
    exit(1)
  return bluroutput

# unpad a minc file. Remove 10 layers of voxels all around the volume
def unpad_minc_file(minc_file, outputname, temp_dir):
  xlength = os.popen("mincinfo -dimlength xspace %s" % minc_file).readlines()[-1]
  ylength = os.popen("mincinfo -dimlength yspace %s" % minc_file).readlines()[-1]
  zlength = os.popen("mincinfo -dimlength zspace %s" % minc_file).readlines()[-1]
  xlength = int(xlength) - 20
  ylength = int(ylength) - 20
  zlength = int(zlength) - 20
  filename = os.path.basename(minc_file)
  basename = filename.replace('.mnc', '')
  tempoutput = "%s/%s_unpadded_temp.mnc" % (temp_dir, basename)
  # minc dimension order: zspace yspace xspace
  command = "mincreshape -clobber -fill -fillvalue 1 -start 10,10,10 -count %s,%s,%s %s %s" \
            % (zlength, ylength, xlength, minc_file, tempoutput)
  os.system(command)
  # set the correct image ranges
  imagemin = float(os.popen("mincstats -quiet -min %s" % tempoutput).readlines()[-1])
  imagemax = float(os.popen("mincstats -quiet -max %s" % tempoutput).readlines()[-1])
  command = "mincreshape -clobber -image_range %s %s %s %s" \
            % (imagemin, imagemax, tempoutput, outputname)
  os.system(command)
  return outputname

###############################################################################
#################################### MAIN #####################################
###############################################################################

if __name__ == "__main__":
  usage = "usage: %prog [options] in_determinant.mnc out_blurred_determinant.mnc"
  parser = OptionParser(usage)
  g_program = sys.argv[0]

  parser.add_option("-b", "--blurkernel", dest="blurkernel",
                    help="Specify the blurring kernel for the determinant file",
                    type="float", default=0.1)
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
  g_tmpdir = "%s/tmp_blur" % g_pwd
  if not os.access(g_tmpdir, 0):
    os.mkdir(g_tmpdir)
  
  # check whether the input file is minc2, if not create a 
  # minc2 version of it
  g_minc2_input = impose_minc2_ness(g_source, g_tmpdir)
 
  # Blurring: pad the determinant file before blurring
  # to prevent artifacts around the outside of the volume and remove
  # the padding from the file when done blurring
  g_padded_determinant = pad_minc_file(g_minc2_input, g_tmpdir)
  # blur the determinant file
  g_blurred_padded_determinant = blur_minc_file(g_padded_determinant,
                                                options.blurkernel,
                                                g_tmpdir)
  # unpad the blurred determinant file
  unpad_minc_file(g_blurred_padded_determinant, g_output, g_tmpdir)
  
  if not options.keeptemp:
    os.system("rm -fr %s" % g_tmpdir)

  print("\ndone!\n\n")
