#include <cuda.h>
#include <cuda_runtime.h>

#include "kernel_proto.cu.h"

void synchronize_cuda();

// textures
typedef texture<float, cudaTextureType1D, cudaReadModeElementType> realw_texture;

// restricted pointers: improves performance on Kepler ~ 10%
// see: http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#restrict
typedef const float* __restrict__ realw_const_p; // otherwise use: //typedef const float* realw_const_p;
typedef float* __restrict__ realw_p; // otherwise use: //typedef float* realw_p;


#define INITIALIZE_OFFSET_CUDA()

#define INIT_OFFSET_CUDA(_buffer_, _offset_)                        \
  if (run_opencl) {                                                 \
    _buffer_##_##_offset_.cuda = mp->_buffer_.cuda + _offset_;      \
  }                                                                 \

#define RELEASE_OFFSET_CUDA(_buffer_, _offset_)

#define TAKE_REF_CUDA(_buffer_)

void print_CUDA_error_if_any(cudaError_t err, int num);

#ifdef USE_OLDER_CUDA4_GPU
#else
  #ifdef USE_TEXTURES_FIELDS
    // forward
    extern realw_texture d_displ_cm_tex;
    extern realw_texture d_accel_cm_tex;

    extern realw_texture d_displ_oc_tex;
    extern realw_texture d_accel_oc_tex;

    extern realw_texture d_displ_ic_tex;
    extern realw_texture d_accel_ic_tex;

    // backward/reconstructed
    extern realw_texture d_b_displ_cm_tex;
    extern realw_texture d_b_accel_cm_tex;

    extern realw_texture d_b_displ_oc_tex;
    extern realw_texture d_b_accel_oc_tex;

    extern realw_texture d_b_displ_ic_tex;
    extern realw_texture d_b_accel_ic_tex;
  #endif

  #ifdef USE_TEXTURES_CONSTANTS
    // hprime
    extern realw_texture d_hprime_xx_tex;
    extern __constant__ size_t d_hprime_xx_tex_offset;
    // weighted hprime
    extern realw_texture d_hprimewgll_xx_tex;
    extern __constant__ size_t d_hprimewgll_xx_tex_offset;
  #endif
#endif
