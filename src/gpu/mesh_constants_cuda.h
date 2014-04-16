#include <cuda.h>
#include <cuda_runtime.h>

#include "kernel_proto.cu.h"

#define INITIALIZE_OFFSET_CUDA()

#define INIT_OFFSET_CUDA(_buffer_, _offset_)                        \
  if (run_opencl) {                                                 \
    _buffer_##_##_offset_.cuda = mp->_buffer_.cuda + _offset_;      \
  }                                                                 \

#define RELEASE_OFFSET_CUDA(_buffer_, _offset_)

#define TAKE_REF_CUDA(_buffer_)

void print_CUDA_error_if_any(cudaError_t err, int num);

