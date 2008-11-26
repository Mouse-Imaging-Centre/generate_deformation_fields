#include <minc2.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <iostream>
#include <math.h>
#include <ParseArgv.h>


void read_open_minc_volume(char* file, mihandle_t* mincfile) {
  if (miopen_volume(file, MI2_OPEN_READ, mincfile) != MI_NOERROR) {
    fprintf(stderr, "Error opening input file: %s.\n", file);
    exit(1);
  }
}


void retrieve_volume_dimensions_sizes(mihandle_t* mincfile, int dimcount, 
                                      midimhandle_t dims[], unsigned int sizes[]) {
  miget_volume_dimensions(*mincfile, MI_DIMCLASS_ANY,
			  MI_DIMATTR_ALL, MI_DIMORDER_FILE,
			  dimcount, dims);
  miget_dimension_sizes(dims, dimcount, sizes);
}


void update_neighbors(unsigned int x, unsigned int y, unsigned int z,
                      double det_diff, double *slabgrid, unsigned int sizes_grid[]) {
  double static_separation = 0.12;
  double vec_length = static_separation * det_diff * 0.02;
  int grid_index;
  
  //update vector dimension neighbors: vec (0,1,2,) = (x,y,z)
  //x-1,y,z   adjust x vector
  grid_index = ((0 * sizes_grid[1] + z) * sizes_grid[2] + y) * sizes_grid[3] + x-1;
  slabgrid[grid_index] -= vec_length;
  
  //x+1,y,z   adjust x vector
  grid_index = ((0 * sizes_grid[1] + z) * sizes_grid[2] + y) * sizes_grid[3] + x+1;
  slabgrid[grid_index] += vec_length;
  
  //x,y-1,z   adjust y vector
  grid_index = ((1 * sizes_grid[1] + z) * sizes_grid[2] + y-1) * sizes_grid[3] + x;
  slabgrid[grid_index] -= vec_length;
  
  //x,y+1,z   adjust y vector
  grid_index = ((1 * sizes_grid[1] + z) * sizes_grid[2] + y+1) * sizes_grid[3] + x;
  slabgrid[grid_index] += vec_length;
  
  //x,y,z-1   adjust z vector
  grid_index = ((2 * sizes_grid[1] + z-1) * sizes_grid[2] + y) * sizes_grid[3] + x;
  slabgrid[grid_index] -= vec_length;
  
  //x-1,y,z   adjust x vector
  grid_index = ((2 * sizes_grid[1] + z+1) * sizes_grid[2] + y) * sizes_grid[3] + x;
  slabgrid[grid_index] += vec_length;
}


/* argument variables */
static int clobber = FALSE;
static double tolerance = 0.02;
/* argument table

  origin of the deformation area
  radius of the deformation sphere
  value of determinant in the deformation area (outside 1, i.e., no change)

*/
static ArgvInfo argTable[] = {
  {NULL, ARGV_HELP, (char *)NULL, (char *)NULL,
    (char*) "General options:"},
  {(char*) "-clobber", ARGV_CONSTANT, (char *)TRUE, (char *)&clobber,
    (char*) "clobber existing files"},
  {NULL, ARGV_HELP, (char *)NULL, (char *)NULL,
    (char*) "\nSpecific options:"},
  {(char*) "-tolerance", ARGV_FLOAT, (char *) 0, (char *) &tolerance,
    (char*) "Specify the tolerance for error between the determinant files (per voxel)."},
  {NULL, ARGV_HELP, (char *)NULL, (char *)NULL, (char*) ""},

  {NULL, ARGV_END, NULL, NULL, NULL}
};


int main(int argc, char **argv) {
  mihandle_t    determinant_v, current_v, grid_v, outgrid_v;
  midimhandle_t dims_det[3], dims_cur[3], dims_grid[4], *dims_new;
  unsigned int  sizes_det[3], sizes_cur[3], sizes_grid[4];
  int           i, /*grid_index,*/ v_index;
  unsigned long start_grid[4], count_grid[4], start_v[3], count_v[3];
  double        *slabgrid, *slabdet, *slabcur;
  double        min, max, det_diff;
  unsigned int  x,y,z;


  // check arguments 
  if(ParseArgv(&argc, argv, argTable, 0) || (argc != 5)){
    std::cerr << "\nUsage: " << argv[0] 
              << " [options] target_determinant.mnc current_determinant.mnc current_grid.mnc output_grid\n"
              << argv[0] << " -help\n\n";
    return 1;
  }


  //open input volumes
  read_open_minc_volume(argv[1], &determinant_v);
  read_open_minc_volume(argv[2], &current_v);
  read_open_minc_volume(argv[3], &grid_v);


  //check whether the output file exists already
  if(access(argv[4], F_OK) == 0 && !clobber){
    std::cerr << "\nError: " << argv[0] << ": " << argv[4]
      << " exists! (use -clobber to overwrite)\n\n"; 
    exit(EXIT_FAILURE);
  }

  
  //get dimension sizes
  retrieve_volume_dimensions_sizes(&determinant_v, 3, dims_det, sizes_det);
  retrieve_volume_dimensions_sizes(&current_v, 3, dims_cur, sizes_cur);
  retrieve_volume_dimensions_sizes(&grid_v, 4, dims_grid, sizes_grid);
  
  
  //allocate dimensions for the output grid
  dims_new = (midimhandle_t*) malloc(sizeof(midimhandle_t) * 4);
  //copy grid dimensions
  for(i = 0; i < 4; i++) {
    micopy_dimension(dims_grid[i], &dims_new[i]);
  }
  //fix the unit attribute of the vector_dimension
  //otherwise HDF5 complains...
  miset_dimension_units(*dims_new, (char*)"mm");
  
  
  start_grid[0] = start_grid[1] = start_grid[2] = start_grid[3] = 0;
  start_v[0] = start_v[1] = start_v[2] = 0;
  for(i = 0; i < 4; i++) {  
    count_grid[i] = (unsigned long) sizes_grid[i];
  }
  for(i = 0; i < 3; i++) {
    count_v[i] = (unsigned long) sizes_det[i];
  }
  
  
  //DETERMINANT: allocate memory for hyperslab; entire volume
  slabdet = (double*) malloc(sizeof(double) * sizes_det[0] * sizes_det[1] * sizes_det[2]);
  if (miget_real_value_hyperslab(determinant_v, MI_TYPE_DOUBLE,
				 start_v, count_v, slabdet) != MI_NOERROR) {
    fprintf(stderr, "\nError getting hyperslab DETERMINANT");
    return(1);
  }
  
  
  //CURRENT: allocate memory for hyperslab; entire volume
  slabcur = (double*) malloc(sizeof(double) * sizes_cur[0] * sizes_cur[1] * sizes_cur[2]);
  if (miget_real_value_hyperslab(current_v, MI_TYPE_DOUBLE,
				 start_v, count_v, slabcur) != MI_NOERROR) {
    fprintf(stderr, "\nError getting hyperslab CURRENT");
    return(1);
  }
  
  
  //GRID: allocate memory for hyperslab; entire volume
  slabgrid = (double*) malloc(sizeof(double) * sizes_grid[0] * sizes_grid[1] * sizes_grid[2] * sizes_grid[3]);
  if (miget_real_value_hyperslab(grid_v, MI_TYPE_DOUBLE,
				 start_grid, count_grid, slabgrid) != MI_NOERROR) {
    fprintf(stderr, "\nError getting hyperslab GRID");
    return(1);
  }
  
  
  min = 100;
  max = -100;
  
  double total_diff = 0;
  
  //MAIN LOOP
  //we'll loop over all voxels in the target determinant file and
  //compare it with the current determinant file
  for(z = 1; z < sizes_det[0]-1; z++) {
    for(y = 1; y < sizes_det[1]-1; y++) {
      for(x = 1; x < sizes_det[2]-1; x++) {
        v_index = (z * sizes_det[1] + y) * sizes_det[2] + x;
        det_diff = slabdet[v_index] - slabcur[v_index];
        if(det_diff < -tolerance || det_diff > tolerance) {
          total_diff += fabs(det_diff);
          update_neighbors(x,y,z,det_diff, slabgrid, sizes_grid);
        }
      }
    }
  }
  
  std::cout << "TOTAL DIFF: " << total_diff << std::endl;
  
  /* create the new volume */
  if (micreate_volume(argv[4], 4, dims_new, MI_TYPE_DOUBLE,
		      MI_CLASS_REAL, NULL, &outgrid_v) != MI_NOERROR) {
    fprintf(stderr, "Error creating new volume\n");
    return(1);
  }
  
  /* create the data for the new volume */
  if (micreate_volume_image(outgrid_v) != MI_NOERROR) {
    fprintf(stderr, "Error creating volume data\n");
    return(1);
  }
  
  miset_volume_range(outgrid_v, 100, -100);
  
  /* write the modified hyperslab to the file */
  if (miset_real_value_hyperslab(outgrid_v, MI_TYPE_DOUBLE,
				 start_grid, count_grid, slabgrid) != MI_NOERROR) {
    fprintf(stderr, "Error setting hyperslab\n");
    return(1);
  }

  free(dims_new);
  free(slabgrid);
  free(slabdet);
  free(slabcur);

  return(0);
}


