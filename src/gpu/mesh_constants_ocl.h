#include <CL/cl.h>

const char* clewErrorString (cl_int error);

#define INITIALIZE_OFFSET_OCL()                     \
  cl_uint buffer_create_type;                       \
  size_t size;                                      \
  cl_buffer_region region_type;

#define INIT_OFFSET_OCL(_buffer_, _offset_)                             \
  if (run_opencl) {                                                     \
    clCheck (clGetMemObjectInfo (mp->_buffer_.ocl, CL_MEM_FLAGS,        \
                                 sizeof(cl_uint),                       \
                                 &buffer_create_type,                   \
                                 NULL));                                \
    clCheck (clGetMemObjectInfo (mp->_buffer_.ocl,                      \
                                 CL_MEM_SIZE ,                          \
                                 sizeof(size_t),                        \
                                 &size,                                 \
                                 NULL));                                \
                                                                        \
    region_type.origin = _offset_ * sizeof(CL_FLOAT);                   \
    region_type.size = size;                                            \
                                                                        \
    _buffer_##_##_offset_.ocl = clCreateSubBuffer (mp->_buffer_.ocl,    \
                                                   buffer_create_type,  \
                                                   CL_BUFFER_CREATE_TYPE_REGION, \
                                                   (void *) &region_type, \
                                                   clck_(&mocl_errcode)); \
  }


#define RELEASE_OFFSET_OCL(_buffer_, _offset_)          \
  if (run_opencl) {                                     \
    clReleaseMemObject(_buffer_##_##_offset_.ocl);      \
  }

extern int mocl_errcode;
static inline cl_int _clCheck(cl_int errcode, const char *file, int line, const char *func) {
  mocl_errcode = errcode;
  if (mocl_errcode != CL_SUCCESS) {
    fprintf (stderr, "Error %d/%s at %s:%d %s\n", mocl_errcode,
             clewErrorString(mocl_errcode),
             file, line, func);
    fflush(NULL);
    exit(1);
  }
  return errcode;
}

#define clCheck(to_check) _clCheck(to_check,__FILE__, __LINE__,  __func__)

#define clck_(var) var); clCheck(*var

#define TAKE_REF_OCL(_buffer_)                                  \
  if (run_opencl) {                                             \
    clCheck(clRetainMemObject(_buffer_.ocl));                   \
  }


// only used #ifdef USE_TEXTURES_FIELDS
extern cl_mem d_displ_cm_tex;
extern cl_mem d_accel_cm_tex;

extern cl_mem d_displ_oc_tex;
extern cl_mem d_accel_oc_tex;

extern cl_mem d_displ_ic_tex;
extern cl_mem d_accel_ic_tex;

// only used #ifdef USE_TEXTURES_CONSTANTS
extern cl_mem d_hprime_xx_cm_tex;
extern cl_mem d_hprime_xx_oc_tex;
extern cl_mem d_hprime_xx_ic_tex;

struct mesh_programs_s {
#undef BOAST_KERNEL
#define BOAST_KERNEL(__kern_name__) cl_program __kern_name__##_program

  #include "kernel_list.h"
};
  
struct mesh_kernels_s {
#undef BOAST_KERNEL
#define BOAST_KERNEL(__kern_name__) cl_kernel __kern_name__
  
  #include "kernel_list.h"
};

extern struct _mesh_opencl {
  struct mesh_programs_s programs;
  struct mesh_kernels_s kernels;
  cl_command_queue command_queue;
  cl_command_queue copy_queue;
  cl_context context;
  cl_device_id device;
  cl_int nb_devices;
} mocl;
