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

#ifndef PREPARE_CONSTANTS_GPU
#define PREPARE_CONSTANTS_GPU

// type of "working" variables
typedef float realw;

#ifdef USE_CUDA
/* note:
   constant arrays when used in other compute_forces_***_cuda.cu routines stay zero,
   constant declaration and cudaMemcpyToSymbol would have to be in the same file...

   extern keyword doesn't work for __constant__ declarations.

   also:
   cudaMemcpyToSymbol("deviceCaseParams", caseParams, sizeof(CaseParams));
   ..
   and compile with -arch=sm_20

   see also: http://stackoverflow.com/questions/4008031/how-to-use-cuda-constant-memory-in-a-programmer-pleasant-way
   doesn't seem to work.

   we could keep arrays separated for acoustic and elastic routines...

   workaround:

   for now, we store pointers with cudaGetSymbolAddress() function calls.
   we pass those pointers in all other compute_forces_..() routines

   in this file, we can use the above constant array declarations without need of the pointers.

*/

// cuda constant arrays
//
// note: we use definition __device__ to use global memory rather than constant memory registers
//          to avoid over-loading registers; this should help increasing the occupancy on the GPU

__device__ realw d_hprime_xx[NGLL2];

__device__ realw d_hprimewgll_xx[NGLL2];

__device__ realw d_wgllwgll_xy[NGLL2];
__device__ realw d_wgllwgll_xz[NGLL2];
__device__ realw d_wgllwgll_yz[NGLL2];

// wgll_cube: needed only for gravity case
__device__ realw d_wgll_cube[NGLL3];

#endif

/*------------------------------------------------------------------------*/
// CONSTANT arrays setup
/*------------------------------------------------------------------------*/




// setup functions
void setConst_hprime_xx (realw *array, Mesh *mp) {
#ifdef USE_OPENCL
  if (run_opencl) {
    cl_int errcode;
    mp->d_hprime_xx.ocl = clCreateBuffer (mocl.context, CL_MEM_READ_ONLY,
                                          NGLL2 * sizeof (realw), NULL, clck_(&errcode));

    clCheck (clEnqueueWriteBuffer (mocl.command_queue, mp->d_hprime_xx.ocl, CL_FALSE, 0,
                                   NGLL2 * sizeof (realw),
                                   array, 0, NULL, NULL));
  }
#endif
#ifdef USE_CUDA
  if (run_cuda) {
    cudaError_t err = cudaMemcpyToSymbol(d_hprime_xx, array, NGLL2*sizeof(realw));
    if (err != cudaSuccess)
    {
      fprintf(stderr, "Error in setConst_hprime_xx: %s\n", cudaGetErrorString(err));
      fprintf(stderr, "The problem is maybe the target architecture: -arch sm_** in src/specfem3D/Makefile\n");
      fprintf(stderr, "Please double-check with your GPU card\n");
      exit(1);
    }

#ifdef USE_OLDER_CUDA4_GPU
    err = cudaGetSymbolAddress((void**)&(mp->d_hprime_xx),"d_hprime_xx");
#else
    err = cudaGetSymbolAddress((void**)&(mp->d_hprime_xx),d_hprime_xx);
#endif
    if(err != cudaSuccess) {
      fprintf(stderr, "Error with d_hprime_xx: %s\n", cudaGetErrorString(err));
      exit(1);
    }
  }
#endif

}

void setConst_hprimewgll_xx (realw *array, Mesh *mp) {
#ifdef USE_OPENCL
  if (run_opencl) {
    cl_int errcode;
    mp->d_hprimewgll_xx.ocl = clCreateBuffer (mocl.context, CL_MEM_READ_ONLY,
                                              NGLL2 * sizeof (realw), NULL, clck_(&errcode));

    clCheck (clEnqueueWriteBuffer (mocl.command_queue, mp->d_hprimewgll_xx.ocl, CL_FALSE, 0,
                                   NGLL2 * sizeof (realw), array, 0, NULL, NULL));
  }
#endif
#ifdef USE_CUDA
  if (run_cuda) {
    cudaError_t err = cudaMemcpyToSymbol(d_hprimewgll_xx, array, NGLL2*sizeof(realw));
    if (err != cudaSuccess)
    {
      fprintf(stderr, "Error in setConst_hprimewgll_xx: %s\n", cudaGetErrorString(err));
      exit(1);
    }

#ifdef USE_OLDER_CUDA4_GPU
    err = cudaGetSymbolAddress((void**)&(mp->d_hprimewgll_xx),"d_hprimewgll_xx");
#else
    err = cudaGetSymbolAddress((void**)&(mp->d_hprimewgll_xx),d_hprimewgll_xx);
#endif
    if(err != cudaSuccess) {
      fprintf(stderr, "Error with d_hprimewgll_xx: %s\n", cudaGetErrorString(err));
      exit(1);
    }
  }
#endif
}

void setConst_wgllwgll_xy (realw *array, Mesh *mp) {
#ifdef USE_OPENCL
  if (run_opencl) {
    cl_int errcode;
    mp->d_wgllwgll_xy.ocl = clCreateBuffer (mocl.context, CL_MEM_READ_ONLY,
                                            NGLL2 * sizeof (realw), NULL, clck_(&errcode));

    clCheck (clEnqueueWriteBuffer (mocl.command_queue, mp->d_wgllwgll_xy.ocl, CL_FALSE, 0,
                                   NGLL2 * sizeof (realw), array, 0, NULL, NULL));
  }
#endif
#ifdef USE_CUDA
  if (run_cuda) {
    cudaError_t err = cudaMemcpyToSymbol(d_wgllwgll_xy, array, NGLL2*sizeof(realw));
    if (err != cudaSuccess)
    {
      fprintf(stderr, "Error in setConst_wgllwgll_xy: %s\n", cudaGetErrorString(err));
      exit(1);
    }

#ifdef USE_OLDER_CUDA4_GPU
    err = cudaGetSymbolAddress((void**)&(mp->d_wgllwgll_xy),"d_wgllwgll_xy");
#else
    err = cudaGetSymbolAddress((void**)&(mp->d_wgllwgll_xy),d_wgllwgll_xy);
#endif
    if(err != cudaSuccess) {
      fprintf(stderr, "Error with d_wgllwgll_xy: %s\n", cudaGetErrorString(err));
      exit(1);
    }
  }
#endif
}

void setConst_wgllwgll_xz (realw *array, Mesh *mp) {
#ifdef USE_OPENCL
  if (run_opencl) {
    cl_int errcode;
    mp->d_wgllwgll_xz.ocl = clCreateBuffer (mocl.context, CL_MEM_READ_ONLY, NGLL2 * sizeof(realw), NULL, clck_(&errcode));

    clCheck (clEnqueueWriteBuffer (mocl.command_queue, mp->d_wgllwgll_xz.ocl, CL_FALSE, 0,
                                   NGLL2 * sizeof (realw), array, 0, NULL, NULL));
  }
#endif
#ifdef USE_CUDA
  if (run_cuda) {
    cudaError_t err = cudaMemcpyToSymbol(d_wgllwgll_xz, array, NGLL2*sizeof(realw));
    if (err != cudaSuccess)
    {
      fprintf(stderr, "Error in  setConst_wgllwgll_xz: %s\n", cudaGetErrorString(err));
      exit(1);
    }

#ifdef USE_OLDER_CUDA4_GPU
    err = cudaGetSymbolAddress((void**)&(mp->d_wgllwgll_xz),"d_wgllwgll_xz");
#else
    err = cudaGetSymbolAddress((void**)&(mp->d_wgllwgll_xz),d_wgllwgll_xz);
#endif
    if(err != cudaSuccess) {
      fprintf(stderr, "Error with d_wgllwgll_xz: %s\n", cudaGetErrorString(err));
      exit(1);
    }
  }
#endif
}

void setConst_wgllwgll_yz (realw *array, Mesh *mp) {
#ifdef USE_OPENCL
  if (run_opencl) {
    cl_int errcode;
    mp->d_wgllwgll_yz.ocl = clCreateBuffer (mocl.context, CL_MEM_READ_ONLY, NGLL2*sizeof(realw), NULL, clck_(&errcode));

    clCheck (clEnqueueWriteBuffer (mocl.command_queue, mp->d_wgllwgll_yz.ocl, CL_FALSE, 0,
                                   NGLL2 * sizeof (realw), array, 0, NULL, NULL));
  }
#endif
#ifdef USE_CUDA
  if (run_cuda) {
    cudaError_t err = cudaMemcpyToSymbol(d_wgllwgll_yz, array, NGLL2*sizeof(realw));
    if (err != cudaSuccess)
    {
      fprintf(stderr, "Error in setConst_wgllwgll_yz: %s\n", cudaGetErrorString(err));
      exit(1);
    }

#ifdef USE_OLDER_CUDA4_GPU
    err = cudaGetSymbolAddress((void**)&(mp->d_wgllwgll_yz),"d_wgllwgll_yz");
#else
    err = cudaGetSymbolAddress((void**)&(mp->d_wgllwgll_yz),d_wgllwgll_yz);
#endif
    if(err != cudaSuccess) {
      fprintf(stderr, "Error with d_wgllwgll_yz: %s\n", cudaGetErrorString(err));
      exit(1);
    }

  }
#endif
}

void setConst_wgll_cube (realw *array, Mesh *mp) {
#ifdef USE_OPENCL
  if (run_opencl) {
    cl_int errcode;
    // wgll_cube: needed only for gravity case

    mp->d_wgll_cube.ocl = clCreateBuffer (mocl.context, CL_MEM_READ_ONLY, NGLL3 * sizeof(realw), NULL, clck_(&errcode));

    clCheck (clEnqueueWriteBuffer (mocl.command_queue, mp->d_wgll_cube.ocl, CL_FALSE, 0,
                                   NGLL3 * sizeof (realw), array, 0, NULL, NULL));
  }
#endif
#ifdef USE_CUDA
  if (run_cuda) {
    cudaError_t err = cudaMemcpyToSymbol(d_wgll_cube, array, NGLL3*sizeof(realw));
    if (err != cudaSuccess)
    {
      fprintf(stderr, "Error in setConst_wgll_cube: %s\n", cudaGetErrorString(err));
      exit(1);
    }

#ifdef USE_OLDER_CUDA4_GPU
    err = cudaGetSymbolAddress((void**)&(mp->d_wgll_cube),"d_wgll_cube");
#else
    err = cudaGetSymbolAddress((void**)&(mp->d_wgll_cube),d_wgll_cube);
#endif
    if(err != cudaSuccess) {
      fprintf(stderr, "Error with d_wgll_cube: %s\n", cudaGetErrorString(err));
      exit(1);
    }
  }
#endif
}

#endif //PREPARE_CONSTANTS_GPU
