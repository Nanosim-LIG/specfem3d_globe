#define NGLL3 125

__global__ void write_seismograms_transfer_from_device_kernel(int* number_receiver_global,
                                                            int* ispec_selected_rec,
                                                            int* ibool,
                                                            realw* station_seismo_field,
                                                            realw* desired_field,
                                                            int nrec_local) {

// vector fields

  int blockID = blockIdx.x + blockIdx.y*gridDim.x;

  if(blockID < nrec_local) {
    int irec = number_receiver_global[blockID]-1;
    int ispec = ispec_selected_rec[irec]-1;
    int iglob = ibool[threadIdx.x + NGLL3*ispec]-1;

    station_seismo_field[3*NGLL3*blockID + 3*threadIdx.x+0] = desired_field[3*iglob];
    station_seismo_field[3*NGLL3*blockID + 3*threadIdx.x+1] = desired_field[3*iglob+1];
    station_seismo_field[3*NGLL3*blockID + 3*threadIdx.x+2] = desired_field[3*iglob+2];
  }
}
