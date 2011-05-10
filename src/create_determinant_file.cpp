#include <minc.h>
#include <minc2.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <volume_io.h>
#include <bicpl.h>
#include <time_stamp.h>
#include <ParseArgv.h>
#include <assert.h>
#include <string.h>
#include <iostream>
#include <climits>


/* Macros used in program */
#define ISSPACE(ch) (isspace((int)ch))
/* Constants used in program */
#define VECTOR_SEPARATOR ','

static int get_arg_vector(char *dst, char *key, char *nextArg);


/* argument variables */
static int verbose = FALSE;
static int clobber = FALSE;
static long origin_deformation[MAX_VAR_DIMS] = {LONG_MIN};
static int sphere_radius = 1;
static double determinant_value = 1;


/* argument table

  origin of the deformation area
  radius of the deformation sphere
  value of determinant in the deformation area (outside 1, i.e., no change)

*/
static ArgvInfo argTable[] = {
  {NULL, ARGV_HELP, (char *)NULL, (char *)NULL,
    (char*) "General options:"},
  {(char*) "-verbose", ARGV_CONSTANT, (char *)TRUE, (char *)&verbose,
    (char*) "print out extra information"},
  {(char*) "-clobber", ARGV_CONSTANT, (char *)TRUE, (char *)&clobber,
    (char*) "clobber existing files"},
  {NULL, ARGV_HELP, (char *)NULL, (char *)NULL,
    (char*) "\nSpecific options:"},
  {(char*) "-center", ARGV_FUNC, (char *) get_arg_vector, (char *) origin_deformation,
    (char*) "Specify the center of the deformation area (x,y,z)."},
  {(char*) "-radius", ARGV_INT, (char *) 0, (char *) &sphere_radius,
    (char*) "Specify the radius of the deformation area (sphere) in voxels."},
  {(char*) "-determinant", ARGV_FLOAT, (char *) 0, (char *) &determinant_value,
    (char*) "Specify the determinant value for the deformation area."},
  {NULL, ARGV_HELP, (char *)NULL, (char *)NULL, (char*) ""},

  {NULL, ARGV_END, NULL, NULL, NULL}
};

int main(int argc, char *argv[]) {
  char *arg_string;
  char *infile;
  char *outfile;
  int sizes[MAX_VAR_DIMS];
  Real original_separation[3];
  Real original_starts[3];
  Volume eval_volume, new_grid;
  General_transform *voxel_to_world;
  STRING *dimnames;
  int i, j, k, circle_radius;
  int v1, v2, v3;
  int yaxisradius;
  progress_struct progress;
  
  arg_string = time_stamp(argc, argv);
  // check arguments 
  if(ParseArgv(&argc, argv, argTable, 0) || (argc != 3)){
    std::cerr << "\nUsage: " << argv[0] 
              << " [options] input.mnc -center x,y,z  determinant_output.mnc\n\n"
              << argv[0] << " -help\n\n";
    return 1;
  }
  
  infile = argv[1];
  outfile = argv[2];
  std::cout << "\nInput file                : " << infile 
            << "\nOutfile                   : "  << outfile
            << "\nCenter of deformation area: " << origin_deformation[0]
            << ',' << origin_deformation[1] << ',' << origin_deformation[2]
            << "\nRadius of sphere          : " << sphere_radius 
            << "\nDeformation value of area : " << determinant_value << std::endl;
  
  // check for the infile and outfile 
  if(access(infile, F_OK) != 0){
    std::cerr << "\nError: " << argv[0] << ": Couldn't find " << infile << "\n\n";
    exit(EXIT_FAILURE);
  }
  if(access(outfile, F_OK) == 0 && !clobber){
    std::cerr << "\nError: " << argv[0] << ": " << outfile 
      << " exists! (use -clobber to overwrite)\n\n"; 
    exit(EXIT_FAILURE);
  }

  //read input volume, get information
  if (input_volume_header_only( infile, 3, NULL, &eval_volume,
      (minc_input_options *) NULL ) != OK ) 
    return( 1 );

  /* get information about the volume */
  get_volume_sizes( eval_volume, sizes );
  voxel_to_world = get_voxel_to_world_transform(eval_volume);
  dimnames = get_volume_dimension_names(eval_volume);
  get_volume_separations(eval_volume, original_separation);
  get_volume_starts(eval_volume, original_starts);
  
  // test whether the radius of the deformation area 
  // works with the volume and the given center
  if((0 > (origin_deformation[0] - sphere_radius)) || 
     (0 > (origin_deformation[1] - sphere_radius)) || 
     (0 > (origin_deformation[2] - sphere_radius)) ||
     (sizes[2] < (origin_deformation[0] + sphere_radius)) || 
     (sizes[1] < (origin_deformation[1] + sphere_radius)) || 
     (sizes[0] < (origin_deformation[2] + sphere_radius))) {
    std::cerr << "\nError: deformation area does not fit in volume:" 
              << "\nRanges of sphere : " << origin_deformation[0] - sphere_radius
              << '-' << origin_deformation[0] + sphere_radius << ", "
              << origin_deformation[1] - sphere_radius
              << '-' << origin_deformation[1] + sphere_radius << ", "
              << origin_deformation[2] - sphere_radius
              << '-' << origin_deformation[2] + sphere_radius 
              << "\nRanges of volume :  0-" << sizes[2]
              << ",  0-" << sizes[1] << ",  0-" << sizes[0] << std::endl << std::endl;
    return 1;
  }
  
  // create the deformation volume
  new_grid = create_volume(3, dimnames, NC_SHORT, FALSE, 0.0, 0.0);
  if(determinant_value < 1) {
    set_volume_real_range(new_grid, determinant_value, 1);
  }
  else {
    set_volume_real_range(new_grid, 1, determinant_value);
  }
  set_volume_sizes(new_grid, sizes);
  set_volume_separations(new_grid, original_separation);
  set_volume_starts(new_grid, original_starts);
  alloc_volume_data(new_grid);
  initialize_progress_report(&progress, FALSE, sizes[0], (char*) "Processing");
  
  // initialize volume; all 1s
  for( v1 = 0;  v1 < sizes[0];  ++v1 ) {
    update_progress_report(&progress, v1 + 1);
    for( v2 = 0;  v2 < sizes[1];  ++v2 ) {
      for( v3 = 0;  v3 < sizes[2];  ++v3 ) {
        set_volume_real_value(new_grid, v1, v2, v3, 0, 0, 1);
      }
    }
  }
  
  // write out a 3D "sphere"
  // first we loop over one of the axis (z)
  for( k = -sphere_radius; k <= sphere_radius; k++) {
    // now we will draw a 2D circle each time
    // with the largest circle when k=0, middle of the sphere
    circle_radius = sphere_radius - abs(k);
    for( i = -circle_radius; i <= circle_radius; i++) {
      // the other axis of the circle
      yaxisradius = circle_radius - abs(i);
      for(j = -yaxisradius; j <= yaxisradius; j++) {
       set_volume_real_value(new_grid, origin_deformation[2] - k, 
                             origin_deformation[1] - j,
                             origin_deformation[0] - i,
                             0, 0, determinant_value);
      }
    }
  }
  
  terminate_progress_report(&progress);
  printf("Outputting volume.\n");
  output_volume(outfile, MI_ORIGINAL_TYPE, TRUE, 0.0, 0.0, 
    new_grid, arg_string, NULL);
  
  return 0;
};



/* ----------------------------- MNI Header -----------------------------------
@NAME       : get_arg_vector
@INPUT      : key - argv key string (-start, -count)
              nextArg - string from which vector should be read
@OUTPUT     : dst - pointer to vector of longs into which values should
                 be written (padded with LONG_MIN)
@RETURNS    : TRUE, since nextArg is used (unless it is NULL)
@DESCRIPTION: Parses a command-line argument into a vector of longs. The
              string should contain at most MAX_VAR_DIMS comma separated 
              integer values (spaces are skipped).
@METHOD     : 
@GLOBALS    : 
@CALLS      : 
@CREATED    : June 10, 1993 (Peter Neelin)
@MODIFIED   : 
---------------------------------------------------------------------------- */
static int get_arg_vector(char *dst, char *key, char *nextArg)
     /* ARGSUSED */
{

   long *vector;
   int nvals, i;
   char *cur, *end, *prev;

   /* Check for following argument */
   if (nextArg == NULL) {
      (void) fprintf(stderr, 
                     "\"%s\" option requires an additional argument\n",
                     key);
      return FALSE;
   }

   /* Get pointer to vector of longs */
   vector = (long *) dst;

   /* Set up pointers to end of string and first non-space character */
   end = nextArg + strlen(nextArg);
   cur = nextArg;
   while (ISSPACE(*cur)) cur++;
   nvals = 0;

   /* Loop through string looking for integers */
   while ((nvals < MAX_VAR_DIMS) && (cur!=end)) {

      /* Get integer */
      prev = cur;
      vector[nvals] = strtol(prev, &cur, 0);
      if ((cur == prev) ||
          !(ISSPACE(*cur) || (*cur == VECTOR_SEPARATOR) || (*cur == '\0'))) {
         (void) fprintf(stderr, 
            "expected vector of integers for \"%s\", but got \"%s\"\n", 
                        key, nextArg);
         exit(EXIT_FAILURE);
      }
      nvals++;

      /* Skip any spaces */
      while (ISSPACE(*cur)) cur++;

      /* Skip an optional comma */
      if (*cur == VECTOR_SEPARATOR) cur++;

   }

   /* Pad with LONG_MIN */
   for (i=nvals; i < MAX_VAR_DIMS; i++) {
      vector[i] = LONG_MIN;
   }

   return TRUE;
}
