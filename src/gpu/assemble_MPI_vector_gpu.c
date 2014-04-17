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

/* ----------------------------------------------------------------------------------------------- */
// MPI transfer
/* ----------------------------------------------------------------------------------------------- */


// prepares and transfers the inter-element edge-nodes to the host to be MPI'd
// (elements on boundary)

extern EXTERN_LANG
void FC_FUNC_(transfer_boun_from_device,
              TRANSFER_BOUN_FROM_DEVICE)(long *Mesh_pointer_f,
                                                 realw *send_accel_buffer,
                                                 int *IREGION,
                                                 int *FORWARD_OR_ADJOINT) {
  TRACE("transfer_boun_from_device");

  int blocksize, size_padded;
  int num_blocks_x, num_blocks_y;

#ifdef USE_OPENCL
  size_t global_work_size[2];
  size_t local_work_size[2];
#endif
#ifdef USE_CUDA
  dim3 grid,threads;
#endif
  int size_mpi_buffer;

  //get mesh pointer out of fortran integer container
  Mesh *mp = (Mesh *) *Mesh_pointer_f;

  // crust/mantle region
  if (*IREGION == IREGION_CRUST_MANTLE) {
    size_mpi_buffer = NDIM*mp->max_nibool_interfaces_cm*mp->num_interfaces_crust_mantle;

    if( size_mpi_buffer > 0 ){

      blocksize = BLOCKSIZE_TRANSFER;
      size_padded = ((int) ceil (((double) mp->max_nibool_interfaces_cm) / ((double) blocksize))) * blocksize;

      get_blocks_xy (size_padded / blocksize, &num_blocks_x, &num_blocks_y);

#ifdef USE_OPENCL
      if (run_opencl) {
        local_work_size[0] = blocksize;
        local_work_size[1] = 1;
        global_work_size[0] = num_blocks_x * blocksize;
        global_work_size[1] = num_blocks_y;

        if (*FORWARD_OR_ADJOINT == 1) {
          clCheck (clSetKernelArg (mocl.kernels.prepare_boundary_accel_on_device, 0, sizeof (gpu_realw_mem), (void *) &mp->d_accel_crust_mantle.ocl));
          clCheck (clSetKernelArg (mocl.kernels.prepare_boundary_accel_on_device, 1, sizeof (gpu_realw_mem), (void *) &mp->d_send_accel_buffer_crust_mantle.ocl));
          clCheck (clSetKernelArg (mocl.kernels.prepare_boundary_accel_on_device, 2, sizeof (int), (void *) &mp->num_interfaces_crust_mantle));
          clCheck (clSetKernelArg (mocl.kernels.prepare_boundary_accel_on_device, 3, sizeof (int), (void *) &mp->max_nibool_interfaces_cm));
          clCheck (clSetKernelArg (mocl.kernels.prepare_boundary_accel_on_device, 4, sizeof (gpu_int_mem), (void *) &mp->d_nibool_interfaces_crust_mantle.ocl));
          clCheck (clSetKernelArg (mocl.kernels.prepare_boundary_accel_on_device, 5, sizeof (gpu_int_mem), (void *) &mp->d_ibool_interfaces_crust_mantle.ocl));

          clCheck (clEnqueueNDRangeKernel (mocl.command_queue, mocl.kernels.prepare_boundary_accel_on_device, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL));

          // copies buffer to CPU
          clCheck (clEnqueueReadBuffer (mocl.command_queue, mp->d_send_accel_buffer_crust_mantle.ocl, CL_TRUE, 0,
                                        NDIM * mp->max_nibool_interfaces_cm * mp->num_interfaces_crust_mantle * sizeof (realw),
                                        send_accel_buffer, 0, NULL, NULL));

        } else if (*FORWARD_OR_ADJOINT == 3) {
          // debug
          DEBUG_BACKWARD_ASSEMBLY ();

          clCheck (clSetKernelArg (mocl.kernels.prepare_boundary_accel_on_device, 0, sizeof (gpu_realw_mem), (void *) &mp->d_b_accel_crust_mantle.ocl));
          clCheck (clSetKernelArg (mocl.kernels.prepare_boundary_accel_on_device, 1, sizeof (gpu_realw_mem), (void *) &mp->d_b_send_accel_buffer_crust_mantle.ocl));
          clCheck (clSetKernelArg (mocl.kernels.prepare_boundary_accel_on_device, 2, sizeof (int), (void *) &mp->num_interfaces_crust_mantle));
          clCheck (clSetKernelArg (mocl.kernels.prepare_boundary_accel_on_device, 3, sizeof (int), (void *) &mp->max_nibool_interfaces_cm));
          clCheck (clSetKernelArg (mocl.kernels.prepare_boundary_accel_on_device, 4, sizeof (gpu_int_mem), (void *) &mp->d_nibool_interfaces_crust_mantle.ocl));
          clCheck (clSetKernelArg (mocl.kernels.prepare_boundary_accel_on_device, 5, sizeof (gpu_int_mem), (void *) &mp->d_ibool_interfaces_crust_mantle.ocl));

          clCheck (clEnqueueNDRangeKernel (mocl.command_queue, mocl.kernels.prepare_boundary_accel_on_device, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL));

          // copies buffer to CPU
          clCheck (clEnqueueReadBuffer (mocl.command_queue, mp->d_send_accel_buffer_crust_mantle.ocl, CL_TRUE, 0,
                                        NDIM*mp->max_nibool_interfaces_cm*mp->num_interfaces_crust_mantle*sizeof (realw),
                                        send_accel_buffer, 0, NULL, NULL));

        }
      }
#endif
#ifdef USE_CUDA
      if (run_cuda) {
        grid = dim3(num_blocks_x,num_blocks_y);
        threads = dim3(blocksize,1,1);

        if(*FORWARD_OR_ADJOINT == 1) {
          prepare_boundary_accel_on_device<<<grid,threads,0,mp->compute_stream>>>(mp->d_accel_crust_mantle.cuda,
                                                             mp->d_send_accel_buffer_crust_mantle.cuda,
                                                             mp->num_interfaces_crust_mantle,
                                                             mp->max_nibool_interfaces_cm,
                                                             mp->d_nibool_interfaces_crust_mantle.cuda,
                                                             mp->d_ibool_interfaces_crust_mantle.cuda);

          // copies buffer to CPU
          if( GPU_ASYNC_COPY ){
            // waits until kernel is finished before starting async memcpy
            cudaStreamSynchronize(mp->compute_stream);
            // copies buffer to CPU
            cudaMemcpyAsync(mp->h_send_accel_buffer_cm,mp->d_send_accel_buffer_crust_mantle.cuda,size_mpi_buffer*sizeof(realw),
                            cudaMemcpyDeviceToHost,mp->copy_stream);
          }else{
            // synchronuous copy
            print_CUDA_error_if_any(cudaMemcpy(send_accel_buffer,mp->d_send_accel_buffer_crust_mantle.cuda,size_mpi_buffer*sizeof(realw),
                                               cudaMemcpyDeviceToHost),41000);
          }
        } else if(*FORWARD_OR_ADJOINT == 3) {
          // debug
          DEBUG_BACKWARD_ASSEMBLY();

          prepare_boundary_accel_on_device<<<grid,threads,0,mp->compute_stream>>>(mp->d_b_accel_crust_mantle.cuda,
                                                             mp->d_b_send_accel_buffer_crust_mantle.cuda,
                                                             mp->num_interfaces_crust_mantle,
                                                             mp->max_nibool_interfaces_cm,
                                                             mp->d_nibool_interfaces_crust_mantle.cuda,
                                                             mp->d_ibool_interfaces_crust_mantle.cuda);
          // copies buffer to CPU
          if (GPU_ASYNC_COPY) {
            // waits until kernel is finished before starting async memcpy
            cudaStreamSynchronize(mp->compute_stream);
            // copies buffer to CPU
            cudaMemcpyAsync(mp->h_b_send_accel_buffer_cm,mp->d_b_send_accel_buffer_crust_mantle.cuda,size_mpi_buffer*sizeof(realw),
                            cudaMemcpyDeviceToHost,mp->copy_stream);
          } else {
            // synchronuous copy
            print_CUDA_error_if_any(cudaMemcpy(send_accel_buffer,mp->d_b_send_accel_buffer_crust_mantle.cuda,size_mpi_buffer*sizeof(realw),
                                               cudaMemcpyDeviceToHost),41001);
            
          }
        }
      }
#endif
    }
  }

  // inner core region
  if (*IREGION == IREGION_INNER_CORE) {
    size_mpi_buffer = NDIM*mp->max_nibool_interfaces_ic*mp->num_interfaces_inner_core;

    if( size_mpi_buffer > 0 ){

      blocksize = BLOCKSIZE_TRANSFER;
      size_padded = ((int) ceil (((double) mp->max_nibool_interfaces_ic) / ((double) blocksize))) * blocksize;

      get_blocks_xy (size_padded / blocksize, &num_blocks_x, &num_blocks_y);

#ifdef USE_OPENCL
      if (run_opencl) {
        if (*FORWARD_OR_ADJOINT == 1) {
          clCheck (clSetKernelArg (mocl.kernels.prepare_boundary_accel_on_device, 0, sizeof (gpu_realw_mem), (void *) &mp->d_accel_inner_core.ocl));
          clCheck (clSetKernelArg (mocl.kernels.prepare_boundary_accel_on_device, 1, sizeof (gpu_realw_mem), (void *) &mp->d_send_accel_buffer_inner_core.ocl));
          clCheck (clSetKernelArg (mocl.kernels.prepare_boundary_accel_on_device, 2, sizeof (int), (void *) &mp->num_interfaces_inner_core));
          clCheck (clSetKernelArg (mocl.kernels.prepare_boundary_accel_on_device, 3, sizeof (int), (void *) &mp->max_nibool_interfaces_ic));
          clCheck (clSetKernelArg (mocl.kernels.prepare_boundary_accel_on_device, 4, sizeof (gpu_int_mem), (void *) &mp->d_nibool_interfaces_inner_core.ocl));
          clCheck (clSetKernelArg (mocl.kernels.prepare_boundary_accel_on_device, 5, sizeof (gpu_int_mem), (void *) &mp->d_ibool_interfaces_inner_core.ocl));

          local_work_size[0] = blocksize;
          local_work_size[1] = 1;
          global_work_size[0] = num_blocks_x * blocksize;
          global_work_size[1] = num_blocks_y;

          clCheck (clEnqueueNDRangeKernel (mocl.command_queue, mocl.kernels.prepare_boundary_accel_on_device, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL));

          // copies buffer to CPU
          clCheck (clEnqueueReadBuffer (mocl.command_queue, mp->d_send_accel_buffer_inner_core.ocl, CL_TRUE, 0,
                                        NDIM * mp->max_nibool_interfaces_ic * mp->num_interfaces_inner_core * sizeof (realw),
                                        send_accel_buffer, 0, NULL, NULL));

        } else if (*FORWARD_OR_ADJOINT == 3) {
          // debug
          DEBUG_BACKWARD_ASSEMBLY ();

          clCheck (clSetKernelArg (mocl.kernels.prepare_boundary_accel_on_device, 0, sizeof (gpu_realw_mem), (void *) &mp->d_b_accel_inner_core.ocl));
          clCheck (clSetKernelArg (mocl.kernels.prepare_boundary_accel_on_device, 1, sizeof (gpu_realw_mem), (void *) &mp->d_b_send_accel_buffer_inner_core.ocl));
          clCheck (clSetKernelArg (mocl.kernels.prepare_boundary_accel_on_device, 2, sizeof (int), (void *) &mp->num_interfaces_inner_core));
          clCheck (clSetKernelArg (mocl.kernels.prepare_boundary_accel_on_device, 3, sizeof (int), (void *) &mp->max_nibool_interfaces_ic));
          clCheck (clSetKernelArg (mocl.kernels.prepare_boundary_accel_on_device, 4, sizeof (gpu_int_mem), (void *) &mp->d_nibool_interfaces_inner_core.ocl));
          clCheck (clSetKernelArg (mocl.kernels.prepare_boundary_accel_on_device, 5, sizeof (gpu_int_mem), (void *) &mp->d_ibool_interfaces_inner_core.ocl));

          local_work_size[0] = blocksize;
          local_work_size[1] = 1;
          global_work_size[0] = num_blocks_x * blocksize;
          global_work_size[1] = num_blocks_y;

          clCheck (clEnqueueNDRangeKernel (mocl.command_queue, mocl.kernels.prepare_boundary_accel_on_device, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL));

          // copies buffer to CPU
          clCheck (clEnqueueReadBuffer (mocl.command_queue, mp->d_b_send_accel_buffer_inner_core.ocl, CL_TRUE, 0,
                                        NDIM * mp->max_nibool_interfaces_ic * mp->num_interfaces_inner_core * sizeof (realw),
                                        send_accel_buffer, 0, NULL, NULL));
        }
      }
#endif
#ifdef USE_CUDA
      if (run_cuda) {
        grid = dim3(num_blocks_x,num_blocks_y);
        threads = dim3(blocksize,1,1);

        if(*FORWARD_OR_ADJOINT == 1) {
          prepare_boundary_accel_on_device<<<grid,threads,0,mp->compute_stream>>>(mp->d_accel_inner_core.cuda,
                                                                                  mp->d_send_accel_buffer_inner_core.cuda,
                                                                                  mp->num_interfaces_inner_core,
                                                                                  mp->max_nibool_interfaces_ic,
                                                                                  mp->d_nibool_interfaces_inner_core.cuda,
                                                                                  mp->d_ibool_interfaces_inner_core.cuda);

          // copies buffer to CPU
          if( GPU_ASYNC_COPY ){
            // waits until kernel is finished before starting async memcpy
            cudaStreamSynchronize(mp->compute_stream);
            // copies buffer to CPU
            cudaMemcpyAsync(mp->h_send_accel_buffer_ic,mp->d_send_accel_buffer_inner_core.cuda,size_mpi_buffer*sizeof(realw),
                            cudaMemcpyDeviceToHost,mp->copy_stream);
          }else{
            // synchronuous copy
            print_CUDA_error_if_any(cudaMemcpy(send_accel_buffer,mp->d_send_accel_buffer_inner_core.cuda,size_mpi_buffer*sizeof(realw),
                                               cudaMemcpyDeviceToHost),41000);
            
          }
        } else if (*FORWARD_OR_ADJOINT == 3) {
          // debug
          DEBUG_BACKWARD_ASSEMBLY();
          
          prepare_boundary_accel_on_device<<<grid,threads,0,mp->compute_stream>>>(mp->d_b_accel_inner_core.cuda,
                                                                                  mp->d_b_send_accel_buffer_inner_core.cuda,
                                                                                  mp->num_interfaces_inner_core,
                                                                                  mp->max_nibool_interfaces_ic,
                                                                                  mp->d_nibool_interfaces_inner_core.cuda,
                                                                                  mp->d_ibool_interfaces_inner_core.cuda);
          // copies buffer to CPU
          if( GPU_ASYNC_COPY ){
            // waits until kernel is finished before starting async memcpy
            cudaStreamSynchronize(mp->compute_stream);
            // copies buffer to CPU
            cudaMemcpyAsync(mp->h_b_send_accel_buffer_ic,mp->d_b_send_accel_buffer_inner_core.cuda,size_mpi_buffer*sizeof(realw),
                            cudaMemcpyDeviceToHost,mp->copy_stream);
          }else{
            // synchronuous copy
            print_CUDA_error_if_any(cudaMemcpy(send_accel_buffer,mp->d_b_send_accel_buffer_inner_core.cuda,size_mpi_buffer*sizeof(realw),
                                               cudaMemcpyDeviceToHost),41001);
          }
        }
      }
#endif
    }
  }
#ifdef ENABLE_VERY_SLOW_ERROR_CHECKING
  exit_on_gpu_error ("transfer_boun_from_device");
#endif
}

// FORWARD_OR_ADJOINT == 1 for accel, and == 3 for b_accel
extern EXTERN_LANG
void FC_FUNC_ (transfer_asmbl_accel_to_device,
               TRANSFER_ASMBL_ACCEL_TO_DEVICE) (long *Mesh_pointer,
                                                realw *buffer_recv_vector,
                                                int *IREGION,
                                                int *FORWARD_OR_ADJOINT) {
  TRACE ("transfer_asmbl_accel_to_device");

  int blocksize, size_padded;
  int num_blocks_x, num_blocks_y;
  int size_mpi_buffer;
  
#ifdef USE_OPENCL
  size_t global_work_size[2];
  size_t local_work_size[2];
#endif
#ifdef USE_CUDA
  dim3 grid,threads;
#endif

  Mesh *mp = (Mesh *) *Mesh_pointer;     //get mesh pointer out of fortran integer container

  // crust/mantle region
  if (*IREGION == IREGION_CRUST_MANTLE) {
    size_mpi_buffer = NDIM*(mp->max_nibool_interfaces_cm)*(mp->num_interfaces_crust_mantle);

    if( size_mpi_buffer > 0 ){
      // assembles values
      blocksize = BLOCKSIZE_TRANSFER;
      size_padded = ((int) ceil (((double) mp->max_nibool_interfaces_cm) / ((double) blocksize))) * blocksize;

      get_blocks_xy (size_padded / blocksize, &num_blocks_x, &num_blocks_y);

#ifdef USE_OPENCL
      if (run_opencl) {
        if (*FORWARD_OR_ADJOINT == 1) {
          // copies vector buffer values to GPU
          clCheck (clEnqueueWriteBuffer (mocl.command_queue, mp->d_send_accel_buffer_crust_mantle.ocl, CL_FALSE, 0,
                                         NDIM * (mp->max_nibool_interfaces_cm) * (mp->num_interfaces_crust_mantle)*sizeof (realw),
                                         buffer_recv_vector, 0, NULL, NULL));
          //assemble forward accel
          clCheck (clSetKernelArg (mocl.kernels.assemble_boundary_accel_on_device, 0, sizeof (gpu_realw_mem), (void *) &mp->d_accel_crust_mantle.ocl));
          clCheck (clSetKernelArg (mocl.kernels.assemble_boundary_accel_on_device, 1, sizeof (gpu_realw_mem), (void *) &mp->d_send_accel_buffer_crust_mantle.ocl));
          clCheck (clSetKernelArg (mocl.kernels.assemble_boundary_accel_on_device, 2, sizeof (int), (void *) &mp->num_interfaces_crust_mantle));
          clCheck (clSetKernelArg (mocl.kernels.assemble_boundary_accel_on_device, 3, sizeof (int), (void *) &mp->max_nibool_interfaces_cm));
          clCheck (clSetKernelArg (mocl.kernels.assemble_boundary_accel_on_device, 4, sizeof (gpu_int_mem), (void *) &mp->d_nibool_interfaces_crust_mantle.ocl));
          clCheck (clSetKernelArg (mocl.kernels.assemble_boundary_accel_on_device, 5, sizeof (gpu_int_mem), (void *) &mp->d_ibool_interfaces_crust_mantle.ocl));

          local_work_size[0] = blocksize;
          local_work_size[1] = 1;
          global_work_size[0] = num_blocks_x * blocksize;
          global_work_size[1] = num_blocks_y;

          clCheck (clEnqueueNDRangeKernel (mocl.command_queue, mocl.kernels.assemble_boundary_accel_on_device, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL));
        } else if (*FORWARD_OR_ADJOINT == 3) {
          // debug
          DEBUG_BACKWARD_ASSEMBLY ();

          // copies vector buffer values to GPU
          clCheck (clEnqueueWriteBuffer (mocl.command_queue, mp->d_b_send_accel_buffer_crust_mantle.ocl, CL_FALSE, 0,
                                         NDIM * (mp->max_nibool_interfaces_cm) * (mp->num_interfaces_crust_mantle) * sizeof (realw),
                                         buffer_recv_vector, 0, NULL, NULL));
          //assemble adjoint accel
          clCheck (clSetKernelArg (mocl.kernels.assemble_boundary_accel_on_device, 0, sizeof (gpu_realw_mem), (void *) &mp->d_b_accel_crust_mantle.ocl));
          clCheck (clSetKernelArg (mocl.kernels.assemble_boundary_accel_on_device, 1, sizeof (gpu_realw_mem), (void *) &mp->d_b_send_accel_buffer_crust_mantle.ocl));
          clCheck (clSetKernelArg (mocl.kernels.assemble_boundary_accel_on_device, 2, sizeof (int), (void *) &mp->num_interfaces_crust_mantle));
          clCheck (clSetKernelArg (mocl.kernels.assemble_boundary_accel_on_device, 3, sizeof (int), (void *) &mp->max_nibool_interfaces_cm));
          clCheck (clSetKernelArg (mocl.kernels.assemble_boundary_accel_on_device, 4, sizeof (gpu_int_mem), (void *) &mp->d_nibool_interfaces_crust_mantle.ocl));
          clCheck (clSetKernelArg (mocl.kernels.assemble_boundary_accel_on_device, 5, sizeof (gpu_int_mem), (void *) &mp->d_ibool_interfaces_crust_mantle.ocl));

          local_work_size[0] = blocksize;
          local_work_size[1] = 1;
          global_work_size[0] = num_blocks_x * blocksize;
          global_work_size[1] = num_blocks_y;

          clCheck (clEnqueueNDRangeKernel (mocl.command_queue, mocl.kernels.assemble_boundary_accel_on_device, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL));
        }
      }
#endif
#ifdef USE_CUDA
      if (run_cuda) {
        grid = dim3(num_blocks_x,num_blocks_y);
        threads = dim3(blocksize,1,1);

        if(*FORWARD_OR_ADJOINT == 1) {
        // asynchronuous copy
        if( GPU_ASYNC_COPY ){
          // Wait until previous copy stream finishes. We assemble while other compute kernels execute.
          cudaStreamSynchronize(mp->copy_stream);
        }else{
          // (cudaMemcpy implicitly synchronizes all other cuda operations)
          // copies vector buffer values to GPU
          print_CUDA_error_if_any(cudaMemcpy(mp->d_send_accel_buffer_crust_mantle.cuda,buffer_recv_vector,size_mpi_buffer*sizeof(realw),
                                             cudaMemcpyHostToDevice),41000);
        }

          //assemble forward accel
          assemble_boundary_accel_on_device<<<grid,threads,0,mp->compute_stream>>>(mp->d_accel_crust_mantle.cuda,
                                                              mp->d_send_accel_buffer_crust_mantle.cuda,
                                                              mp->num_interfaces_crust_mantle,
                                                              mp->max_nibool_interfaces_cm,
                                                              mp->d_nibool_interfaces_crust_mantle.cuda,
                                                              mp->d_ibool_interfaces_crust_mantle.cuda);
        }
        else if(*FORWARD_OR_ADJOINT == 3) {
          // debug
          DEBUG_BACKWARD_ASSEMBLY();

        // asynchronuous copy
        if( GPU_ASYNC_COPY ){
          // Wait until previous copy stream finishes. We assemble while other compute kernels execute.
          cudaStreamSynchronize(mp->copy_stream);
        }else{
          // (cudaMemcpy implicitly synchronizes all other cuda operations)
          // copies vector buffer values to GPU
          print_CUDA_error_if_any(cudaMemcpy(mp->d_b_send_accel_buffer_crust_mantle.cuda, buffer_recv_vector,
                                             NDIM*(mp->max_nibool_interfaces_cm)*(mp->num_interfaces_crust_mantle)*sizeof(realw),
                                             cudaMemcpyHostToDevice),41000);
        }

          //assemble adjoint accel
          assemble_boundary_accel_on_device<<<grid,threads,0,mp->compute_stream>>>(mp->d_b_accel_crust_mantle.cuda,
                                                              mp->d_b_send_accel_buffer_crust_mantle.cuda,
                                                              mp->num_interfaces_crust_mantle,
                                                              mp->max_nibool_interfaces_cm,
                                                              mp->d_nibool_interfaces_crust_mantle.cuda,
                                                              mp->d_ibool_interfaces_crust_mantle.cuda);
        }
      }
#endif
    }
  }

#ifdef ENABLE_VERY_SLOW_ERROR_CHECKING
  exit_on_gpu_error ("transfer_asmbl_accel_to_device in crust_mantle");
#endif

  // inner core region
  if (*IREGION == IREGION_INNER_CORE) {
    size_mpi_buffer = NDIM*(mp->max_nibool_interfaces_ic)*(mp->num_interfaces_inner_core);

    if( size_mpi_buffer > 0 ){
      // assembles values
      blocksize = BLOCKSIZE_TRANSFER;
      size_padded = ((int) ceil (((double) mp->max_nibool_interfaces_ic) / ((double) blocksize))) * blocksize;

      get_blocks_xy (size_padded / blocksize, &num_blocks_x, &num_blocks_y);

#ifdef USE_OPENCL
      if (run_opencl) {
        if (*FORWARD_OR_ADJOINT == 1) {
          // copies buffer values to GPU
          clCheck (clEnqueueWriteBuffer (mocl.command_queue, mp->d_send_accel_buffer_inner_core.ocl, CL_FALSE, 0,
                                         NDIM * (mp->max_nibool_interfaces_ic) * (mp->num_interfaces_inner_core)*sizeof (realw),
                                         buffer_recv_vector, 0, NULL, NULL));

          //assemble forward accel
          clCheck (clSetKernelArg (mocl.kernels.assemble_boundary_accel_on_device, 0, sizeof (gpu_realw_mem), (void *) &mp->d_accel_inner_core.ocl));
          clCheck (clSetKernelArg (mocl.kernels.assemble_boundary_accel_on_device, 1, sizeof (gpu_realw_mem), (void *) &mp->d_send_accel_buffer_inner_core.ocl));
          clCheck (clSetKernelArg (mocl.kernels.assemble_boundary_accel_on_device, 2, sizeof (int), (void *) &mp->num_interfaces_inner_core));
          clCheck (clSetKernelArg (mocl.kernels.assemble_boundary_accel_on_device, 3, sizeof (int), (void *) &mp->max_nibool_interfaces_ic));
          clCheck (clSetKernelArg (mocl.kernels.assemble_boundary_accel_on_device, 4, sizeof (gpu_int_mem), (void *) &mp->d_nibool_interfaces_inner_core.ocl));
          clCheck (clSetKernelArg (mocl.kernels.assemble_boundary_accel_on_device, 5, sizeof (gpu_int_mem), (void *) &mp->d_ibool_interfaces_inner_core.ocl));

          local_work_size[0] = blocksize;
          local_work_size[1] = 1;
          global_work_size[0] = num_blocks_x * blocksize;
          global_work_size[1] = num_blocks_y;

          clCheck (clEnqueueNDRangeKernel (mocl.command_queue, mocl.kernels.assemble_boundary_accel_on_device, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL));
        } else if (*FORWARD_OR_ADJOINT == 3) {
          // debug
          DEBUG_BACKWARD_ASSEMBLY ();

          // copies buffer values to GPU
          clCheck (clEnqueueWriteBuffer (mocl.command_queue, mp->d_b_send_accel_buffer_inner_core.ocl, CL_FALSE, 0,
                                         NDIM * (mp->max_nibool_interfaces_ic) * (mp->num_interfaces_inner_core) * sizeof (realw),
                                         buffer_recv_vector, 0, NULL, NULL));
          //assemble adjoint accel
          clCheck (clSetKernelArg (mocl.kernels.assemble_boundary_accel_on_device, 0, sizeof (gpu_realw_mem), (void *) &mp->d_b_accel_inner_core.ocl));
          clCheck (clSetKernelArg (mocl.kernels.assemble_boundary_accel_on_device, 1, sizeof (gpu_realw_mem), (void *) &mp->d_b_send_accel_buffer_inner_core.ocl));
          clCheck (clSetKernelArg (mocl.kernels.assemble_boundary_accel_on_device, 2, sizeof (int), (void *) &mp->num_interfaces_inner_core));
          clCheck (clSetKernelArg (mocl.kernels.assemble_boundary_accel_on_device, 3, sizeof (int), (void *) &mp->max_nibool_interfaces_ic));
          clCheck (clSetKernelArg (mocl.kernels.assemble_boundary_accel_on_device, 4, sizeof (gpu_int_mem), (void *) &mp->d_nibool_interfaces_inner_core.ocl));
          clCheck (clSetKernelArg (mocl.kernels.assemble_boundary_accel_on_device, 5, sizeof (gpu_int_mem), (void *) &mp->d_ibool_interfaces_inner_core.ocl));

          local_work_size[0] = blocksize;
          local_work_size[1] = 1;
          global_work_size[0] = num_blocks_x * blocksize;
          global_work_size[1] = num_blocks_y;

          clCheck (clEnqueueNDRangeKernel (mocl.command_queue, mocl.kernels.assemble_boundary_accel_on_device, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL));
        }
      }
#endif
#ifdef USE_CUDA
      if (run_cuda) {
        grid = dim3(num_blocks_x,num_blocks_y);
        threads = dim3(blocksize,1,1);

        if(*FORWARD_OR_ADJOINT == 1) {
        if( GPU_ASYNC_COPY ){
          // Wait until previous copy stream finishes. We assemble while other compute kernels execute.
          cudaStreamSynchronize(mp->copy_stream);
        }else{
          // (cudaMemcpy implicitly synchronizes all other cuda operations)
          // copies buffer values to GPU
          print_CUDA_error_if_any(cudaMemcpy(mp->d_send_accel_buffer_inner_core.cuda,buffer_recv_vector,size_mpi_buffer*sizeof(realw),
                                             cudaMemcpyHostToDevice),41001);
        }

          //assemble forward accel
          assemble_boundary_accel_on_device<<<grid,threads,0,mp->compute_stream>>>(mp->d_accel_inner_core.cuda,
                                                              mp->d_send_accel_buffer_inner_core.cuda,
                                                              mp->num_interfaces_inner_core,
                                                              mp->max_nibool_interfaces_ic,
                                                              mp->d_nibool_interfaces_inner_core.cuda,
                                                              mp->d_ibool_interfaces_inner_core.cuda);
        }
        else if(*FORWARD_OR_ADJOINT == 3) {
          // debug
          DEBUG_BACKWARD_ASSEMBLY();

        if( GPU_ASYNC_COPY ){
          // Wait until previous copy stream finishes. We assemble while other compute kernels execute.
          cudaStreamSynchronize(mp->copy_stream);
        }else{
          // (cudaMemcpy implicitly synchronizes all other cuda operations)
          // copies buffer values to GPU
          print_CUDA_error_if_any(cudaMemcpy(mp->d_b_send_accel_buffer_inner_core.cuda,buffer_recv_vector,size_mpi_buffer*sizeof(realw),
                                             cudaMemcpyHostToDevice),41001);
        }

          //assemble adjoint accel
          assemble_boundary_accel_on_device<<<grid,threads,0,mp->compute_stream>>>(mp->d_b_accel_inner_core.cuda,
                                                              mp->d_b_send_accel_buffer_inner_core.cuda,
                                                              mp->num_interfaces_inner_core,
                                                              mp->max_nibool_interfaces_ic,
                                                              mp->d_nibool_interfaces_inner_core.cuda,
                                                              mp->d_ibool_interfaces_inner_core.cuda);
        }
      }
#endif
    }
  }

#ifdef ENABLE_VERY_SLOW_ERROR_CHECKING
  exit_on_gpu_error ("transfer_asmbl_accel_to_device in inner_core");
#endif
}
