#define HAVE_MINC1 1
#define HAVE_MINC2 1

#include <minc2.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <volume_io.h>
#include <bicpl.h>
#include <ParseArgv.h>
#include <time_stamp.h>
#include <assert.h>
#include <string.h>
#include <iostream>
#include <minc.h>

// argument parsing defaults 
static int verbose = FALSE;
static int clobber = FALSE;

// argument table 
static ArgvInfo argTable[] = {
  {NULL, ARGV_HELP, (char *)NULL, (char *)NULL,
    (char*) "General options:"},
  {(char*) "-verbose", ARGV_CONSTANT, (char *)TRUE, (char *)&verbose,
    (char*) "print out extra information"},
  {(char*) "-clobber", ARGV_CONSTANT, (char *)TRUE, (char *)&clobber,
    (char*) "clobber existing files"},
  {NULL, ARGV_HELP, (char *)NULL, (char *)NULL, (char*) ""},
  {NULL, ARGV_END, NULL, NULL, NULL}
};

int main(int argc, char *argv[]) {
  int i;
  int v1, v2, v3;
  char *arg_string;
  char *infile;
  char *outfile;
  int sizes[MAX_VAR_DIMS], grid_sizes[4];
  VIO_Real original_separation[3], grid_separation[4];
  VIO_Real original_starts[3], grid_starts[4];
  VIO_STR *dimnames, dimnames_grid[4];
  VIO_Volume eval_volume, new_grid;
  VIO_General_transform *voxel_to_world;
  VIO_progress_struct progress;

  arg_string = time_stamp(argc, argv);
  
  // check arguments 
  if(ParseArgv(&argc, argv, argTable, 0) || (argc != 3)){
    std::cerr << "\nUsage: " << argv[0] 
              << " [options] input.mnc  output_grid.mnc\n\n"
              << argv[0] << " -help\n\n";
    return 1;
  }

  infile = argv[1];
  outfile = argv[2];

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
      (minc_input_options *) NULL ) != VIO_OK ) 
    return( 1 );

  /* get information about the volume */
  get_volume_sizes( eval_volume, sizes );
  voxel_to_world = get_voxel_to_world_transform(eval_volume);
  dimnames = get_volume_dimension_names(eval_volume);
  get_volume_separations(eval_volume, original_separation);
  get_volume_starts(eval_volume, original_starts);

  // create new 4D volume, last three dims same as other volume,
  // first dimension being the vector dimension.
  for(i=1; i < 4; i++) {
    dimnames_grid[i] = dimnames[i-1];
    grid_separation[i] = original_separation[i-1];
    grid_sizes[i] = sizes[i-1];
    grid_starts[i] = original_starts[i-1];
  }
  dimnames_grid[0] = (char*) "vector_dimension";
  grid_sizes[0] = 3;
  grid_separation[0] = 1;
  grid_starts[0] = 0;
  
  new_grid = create_volume(4, dimnames_grid, NC_DOUBLE, FALSE, 0.0, 0.0);
  set_volume_real_range(new_grid, -0.0, 0.0);
  set_volume_sizes(new_grid, grid_sizes);
  set_volume_separations(new_grid, grid_separation);
  set_volume_starts(new_grid, grid_starts);
  
  alloc_volume_data(new_grid);
  
  initialize_progress_report(&progress, FALSE, sizes[0], (char*) "Processing");
  
  //initialize volume
  for( v1 = 0;  v1 < sizes[0];  ++v1 ) {
    update_progress_report(&progress, v1 + 1);
    for( v2 = 0;  v2 < sizes[1];  ++v2 ) {
      for( v3 = 0;  v3 < sizes[2];  ++v3 ) {
        for(i=0; i < 3; i++) {
          set_volume_real_value(new_grid, i, v1, v2, v3, 0.0, 0.0);
        }
      }
    }
  }

  terminate_progress_report(&progress);

  printf("Outputting volume.\n");

  output_volume(outfile, MI_ORIGINAL_TYPE, TRUE, 0.0, 0.0, 
    new_grid, arg_string, NULL);

  return 0;
}
