#define NGLL3 125

__global__ void write_seismograms_transfer_scalar_from_device_kernel(int* number_receiver_global,
                                                                     int* ispec_selected_rec,
                                                                     int* ibool,
                                                                     realw* station_seismo_field,
                                                                     realw* desired_field,
                                                                     int nrec_local) {

// scalar fields

  int blockID = blockIdx.x + blockIdx.y*gridDim.x;

  if(blockID < nrec_local) {
    int irec = number_receiver_global[blockID]-1;
    int ispec = ispec_selected_rec[irec]-1;
    int iglob = ibool[threadIdx.x + NGLL3*ispec]-1;

    station_seismo_field[NGLL3*blockID + threadIdx.x] = desired_field[iglob];
  }
}
