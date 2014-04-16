/*
  !=====================================================================
  !
  !               S p e c f e m 3 D  V e r s i o n  2 . 0
  !               ---------------------------------------
  !
  !          Main authors: Dimitri Komatitsch and Jeroen Tromp
  !    Princeton University, USA and University of Pau / CNRS / INRIA
  ! (c) Princeton University / California Institute of Technology and University of Pau / CNRS / INRIA
  !                            August 2013
  !
  ! This program is free software; you can redistribute it and/or modify
  ! it under the terms of the GNU General Public License as published by
  ! the Free Software Foundation; either version 2 of the License, or
  ! (at your option) any later version.
  !
  ! This program is distributed in the hope that it will be useful,
  ! but WITHOUT ANY WARRANTY; without even the implied warranty of
  ! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  ! GNU General Public License for more details.
  !
  ! You should have received a copy of the GNU General Public License along
  ! with this program; if not, write to the Free Software Foundation, Inc.,
  ! 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
  !
  !=====================================================================
*/

#include "mesh_constants_gpu.h"

/*----------------------------------------------------------------------------------------------- */

// prepares a device array with with all inter-element edge-nodes -- this
// is followed by a memcpy and MPI operations

/*----------------------------------------------------------------------------------------------- */

// prepares and transfers the inter-element edge-nodes to the host to be MPI'd
extern EXTERN_LANG
void FC_FUNC_ (transfer_boun_pot_from_device,
               TRANSFER_BOUN_POT_FROM_DEVICE) (long *Mesh_pointer_f,
                                               realw *send_potential_dot_dot_buffer,
                                               int *FORWARD_OR_ADJOINT) {

  TRACE ("transfer_boun_pot_from_device");

  Mesh *mp = (Mesh *) *Mesh_pointer_f;   //get mesh pointer out of fortran integer container

  // checks if anything to do
  if (mp->num_interfaces_outer_core == 0)
    return;

  int blocksize = BLOCKSIZE_TRANSFER;
  int size_padded = ((int) ceil (((double) mp->max_nibool_interfaces_oc) / ((double) blocksize))) * blocksize;

  int num_blocks_x, num_blocks_y;
  get_blocks_xy (size_padded/blocksize, &num_blocks_x, &num_blocks_y);

#ifdef USE_OPENCL
  if (run_opencl) {
    size_t global_work_size[2];
    size_t local_work_size[2];

    if (*FORWARD_OR_ADJOINT == 1) {
      clCheck (clSetKernelArg (mocl.kernels.prepare_boundary_potential_on_device, 0, sizeof (gpu_realw_mem), (void *) &mp->d_accel_outer_core));
      clCheck (clSetKernelArg (mocl.kernels.prepare_boundary_potential_on_device, 1, sizeof (gpu_realw_mem), (void *) &mp->d_send_accel_buffer_outer_core));
      clCheck (clSetKernelArg (mocl.kernels.prepare_boundary_potential_on_device, 2, sizeof (int), (void *) &mp->num_interfaces_outer_core));
      clCheck (clSetKernelArg (mocl.kernels.prepare_boundary_potential_on_device, 3, sizeof (int), (void *) &mp->max_nibool_interfaces_oc));
      clCheck (clSetKernelArg (mocl.kernels.prepare_boundary_potential_on_device, 4, sizeof (gpu_int_mem), (void *) &mp->d_nibool_interfaces_outer_core));
      clCheck (clSetKernelArg (mocl.kernels.prepare_boundary_potential_on_device, 5, sizeof (gpu_int_mem), (void *) &mp->d_ibool_interfaces_outer_core));

      local_work_size[0] = blocksize;
      local_work_size[1] = 1;
      global_work_size[0] = num_blocks_x * blocksize;
      global_work_size[1] = num_blocks_y;

      clCheck (clEnqueueNDRangeKernel (mocl.command_queue, mocl.kernels.prepare_boundary_potential_on_device, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL));


      clCheck (clEnqueueReadBuffer (mocl.command_queue, mp->d_send_accel_buffer_outer_core.ocl, CL_TRUE, 0,
                                    mp->max_nibool_interfaces_oc * mp->num_interfaces_outer_core * sizeof (realw),
                                    send_potential_dot_dot_buffer, 0, NULL, NULL));
    }
    else if (*FORWARD_OR_ADJOINT == 3) {
      // debug
      DEBUG_BACKWARD_ASSEMBLY ();

      clCheck (clSetKernelArg (mocl.kernels.prepare_boundary_potential_on_device, 0, sizeof (gpu_realw_mem), (void *) &mp->d_b_accel_outer_core));
      clCheck (clSetKernelArg (mocl.kernels.prepare_boundary_potential_on_device, 1, sizeof (gpu_realw_mem), (void *) &mp->d_b_send_accel_buffer_outer_core));
      clCheck (clSetKernelArg (mocl.kernels.prepare_boundary_potential_on_device, 2, sizeof (int), (void *) &mp->num_interfaces_outer_core));
      clCheck (clSetKernelArg (mocl.kernels.prepare_boundary_potential_on_device, 3, sizeof (int), (void *) &mp->max_nibool_interfaces_oc));
      clCheck (clSetKernelArg (mocl.kernels.prepare_boundary_potential_on_device, 4, sizeof (gpu_int_mem), (void *) &mp->d_nibool_interfaces_outer_core));
      clCheck (clSetKernelArg (mocl.kernels.prepare_boundary_potential_on_device, 5, sizeof (gpu_int_mem), (void *) &mp->d_ibool_interfaces_outer_core));

      local_work_size[0] = blocksize;
      local_work_size[1] = 1;
      global_work_size[0] = num_blocks_x * blocksize;
      global_work_size[1] = num_blocks_y;

      clCheck (clEnqueueNDRangeKernel (mocl.command_queue, mocl.kernels.prepare_boundary_potential_on_device, 2, NULL,
                                       global_work_size, local_work_size, 0, NULL, NULL));

      clCheck (clEnqueueReadBuffer (mocl.command_queue, mp->d_b_send_accel_buffer_outer_core.ocl, CL_TRUE, 0,
                                    mp->max_nibool_interfaces_oc * mp->num_interfaces_outer_core * sizeof (realw),
                                    send_potential_dot_dot_buffer, 0, NULL, NULL));
    }
  }
#endif
#ifdef USE_CUDA
  if (run_cuda) {
    dim3 grid(num_blocks_x,num_blocks_y);
    dim3 threads(blocksize,1,1);

    if(*FORWARD_OR_ADJOINT == 1) {
      prepare_boundary_potential_on_device<<<grid,threads>>>(mp->d_accel_outer_core.cuda,
                                                             mp->d_send_accel_buffer_outer_core.cuda,
                                                             mp->num_interfaces_outer_core,
                                                             mp->max_nibool_interfaces_oc,
                                                             mp->d_nibool_interfaces_outer_core.cuda,
                                                             mp->d_ibool_interfaces_outer_core.cuda);

      print_CUDA_error_if_any(cudaMemcpy(send_potential_dot_dot_buffer,mp->d_send_accel_buffer_outer_core.cuda,
                                         (mp->max_nibool_interfaces_oc)*(mp->num_interfaces_outer_core)*sizeof(realw),
                                         cudaMemcpyDeviceToHost),98000);

    }
    else if(*FORWARD_OR_ADJOINT == 3) {
      // debug
      DEBUG_BACKWARD_ASSEMBLY();

      prepare_boundary_potential_on_device<<<grid,threads>>>(mp->d_b_accel_outer_core.cuda,
                                                             mp->d_b_send_accel_buffer_outer_core.cuda,
                                                             mp->num_interfaces_outer_core,
                                                             mp->max_nibool_interfaces_oc,
                                                             mp->d_nibool_interfaces_outer_core.cuda,
                                                             mp->d_ibool_interfaces_outer_core.cuda);

      print_CUDA_error_if_any(cudaMemcpy(send_potential_dot_dot_buffer,mp->d_b_send_accel_buffer_outer_core.cuda,
                                         (mp->max_nibool_interfaces_oc)*(mp->num_interfaces_outer_core)*sizeof(realw),
                                         cudaMemcpyDeviceToHost),98001);

    }
  }
#endif
#ifdef ENABLE_VERY_SLOW_ERROR_CHECKING
  exit_on_gpu_error ("prepare_boundary_kernel");
#endif
}

extern EXTERN_LANG
void FC_FUNC_ (transfer_asmbl_pot_to_device,
               TRANSFER_ASMBL_POT_TO_DEVICE) (long *Mesh_pointer,
                                              realw *buffer_recv_scalar,
                                              int *FORWARD_OR_ADJOINT) {

  TRACE ("transfer_asmbl_pot_to_device");

  //get mesh pointer out of fortran integer container
  Mesh *mp = (Mesh *) (*Mesh_pointer);

  // checks if anything to do
  if (mp->num_interfaces_outer_core == 0)
    return;

  // assembles on GPU
  int blocksize = BLOCKSIZE_TRANSFER;
  int size_padded = ((int) ceil (((double) mp->max_nibool_interfaces_oc) / ((double) blocksize))) * blocksize;

  int num_blocks_x, num_blocks_y;
  get_blocks_xy (size_padded / blocksize, &num_blocks_x, &num_blocks_y);

#ifdef USE_OPENCL
  if (run_opencl) {
    size_t global_work_size[2];
    size_t local_work_size[2];

    if (*FORWARD_OR_ADJOINT == 1) {
      // copies scalar buffer onto GPU
      clCheck (clEnqueueWriteBuffer (mocl.command_queue, mp->d_send_accel_buffer_outer_core.ocl, CL_FALSE, 0,
                                     mp->max_nibool_interfaces_oc * mp->num_interfaces_outer_core * sizeof (realw),
                                     buffer_recv_scalar, 0, NULL, NULL));
      //assemble forward field

      clCheck (clSetKernelArg (mocl.kernels.assemble_boundary_potential_on_device, 0, sizeof (gpu_realw_mem), (void *) &mp->d_accel_outer_core));
      clCheck (clSetKernelArg (mocl.kernels.assemble_boundary_potential_on_device, 1, sizeof (gpu_realw_mem), (void *) &mp->d_send_accel_buffer_outer_core));
      clCheck (clSetKernelArg (mocl.kernels.assemble_boundary_potential_on_device, 2, sizeof (int), (void *) &mp->num_interfaces_outer_core));
      clCheck (clSetKernelArg (mocl.kernels.assemble_boundary_potential_on_device, 3, sizeof (int), (void *) &mp->max_nibool_interfaces_oc));
      clCheck (clSetKernelArg (mocl.kernels.assemble_boundary_potential_on_device, 4, sizeof (gpu_int_mem), (void *) &mp->d_nibool_interfaces_outer_core));
      clCheck (clSetKernelArg (mocl.kernels.assemble_boundary_potential_on_device, 5, sizeof (gpu_int_mem), (void *) &mp->d_ibool_interfaces_outer_core));

      local_work_size[0] = blocksize;
      local_work_size[1] = 1;
      global_work_size[0] = num_blocks_x * blocksize;
      global_work_size[1] = num_blocks_y;

      clCheck (clEnqueueNDRangeKernel (mocl.command_queue, mocl.kernels.assemble_boundary_potential_on_device, 2, NULL,
                                       global_work_size, local_work_size, 0, NULL, NULL));
    }
    else if (*FORWARD_OR_ADJOINT == 3) {
      // debug
      DEBUG_BACKWARD_ASSEMBLY ();

      // copies scalar buffer onto GPU
      clCheck (clEnqueueWriteBuffer (mocl.command_queue, mp->d_b_send_accel_buffer_outer_core.ocl, CL_FALSE, 0,
                                     mp->max_nibool_interfaces_oc * mp->num_interfaces_outer_core * sizeof (realw),
                                     buffer_recv_scalar, 0, NULL, NULL));
      //assemble reconstructed/backward field
      clCheck (clSetKernelArg (mocl.kernels.assemble_boundary_potential_on_device, 0, sizeof (gpu_realw_mem), (void *) &mp->d_b_accel_outer_core));
      clCheck (clSetKernelArg (mocl.kernels.assemble_boundary_potential_on_device, 1, sizeof (gpu_realw_mem), (void *) &mp->d_b_send_accel_buffer_outer_core));
      clCheck (clSetKernelArg (mocl.kernels.assemble_boundary_potential_on_device, 2, sizeof (int), (void *) &mp->num_interfaces_outer_core));
      clCheck (clSetKernelArg (mocl.kernels.assemble_boundary_potential_on_device, 3, sizeof (int), (void *) &mp->max_nibool_interfaces_oc));
      clCheck (clSetKernelArg (mocl.kernels.assemble_boundary_potential_on_device, 4, sizeof (gpu_int_mem), (void *) &mp->d_nibool_interfaces_outer_core));
      clCheck (clSetKernelArg (mocl.kernels.assemble_boundary_potential_on_device, 5, sizeof (gpu_int_mem), (void *) &mp->d_ibool_interfaces_outer_core));

      local_work_size[0] = blocksize;
      local_work_size[1] = 1;
      global_work_size[0] = num_blocks_x * blocksize;
      global_work_size[1] = num_blocks_y;

      clCheck (clEnqueueNDRangeKernel (mocl.command_queue, mocl.kernels.assemble_boundary_potential_on_device, 2, NULL,
                                       global_work_size, local_work_size, 0, NULL, NULL));
    }
  }
#endif
#ifdef USE_CUDA
  if (run_cuda) {
    dim3 grid(num_blocks_x,num_blocks_y);
    dim3 threads(blocksize,1,1);

    if(*FORWARD_OR_ADJOINT == 1) {
      // copies scalar buffer onto GPU
      print_CUDA_error_if_any(cudaMemcpy(mp->d_send_accel_buffer_outer_core.cuda, buffer_recv_scalar,
                                         (mp->max_nibool_interfaces_oc)*(mp->num_interfaces_outer_core)*sizeof(realw),
                                         cudaMemcpyHostToDevice),99000);

      //assemble forward field
      assemble_boundary_potential_on_device<<<grid,threads>>>(mp->d_accel_outer_core.cuda,
                                                              mp->d_send_accel_buffer_outer_core.cuda,
                                                              mp->num_interfaces_outer_core,
                                                              mp->max_nibool_interfaces_oc,
                                                              mp->d_nibool_interfaces_outer_core.cuda,
                                                              mp->d_ibool_interfaces_outer_core.cuda);
    }
    else if(*FORWARD_OR_ADJOINT == 3) {
      // debug
      DEBUG_BACKWARD_ASSEMBLY();

      // copies scalar buffer onto GPU
      print_CUDA_error_if_any(cudaMemcpy(mp->d_b_send_accel_buffer_outer_core.cuda, buffer_recv_scalar,
                                         (mp->max_nibool_interfaces_oc)*(mp->num_interfaces_outer_core)*sizeof(realw),
                                         cudaMemcpyHostToDevice),99001);

      //assemble reconstructed/backward field
      assemble_boundary_potential_on_device<<<grid,threads>>>(mp->d_b_accel_outer_core.cuda,
                                                              mp->d_b_send_accel_buffer_outer_core.cuda,
                                                              mp->num_interfaces_outer_core,
                                                              mp->max_nibool_interfaces_oc,
                                                              mp->d_nibool_interfaces_outer_core.cuda,
                                                              mp->d_ibool_interfaces_outer_core.cuda);
    }
  }
#endif

#ifdef ENABLE_VERY_SLOW_ERROR_CHECKING
  //double end_time = get_time ();
  //printf ("Elapsed time: %e\n", end_time-start_time);
  exit_on_gpu_error ("transfer_asmbl_pot_to_device");
#endif
}
