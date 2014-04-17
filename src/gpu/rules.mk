#=====================================================================
#
#          S p e c f e m 3 D  G l o b e  V e r s i o n  6 . 0
#          --------------------------------------------------
#
#          Main authors: Dimitri Komatitsch and Jeroen Tromp
#                        Princeton University, USA
#             and University of Pau / CNRS / INRIA, France
# (c) Princeton University / California Institute of Technology and University of Pau / CNRS / INRIA
#                            August 2013
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
#=====================================================================

#######################################

gpu_OBJECTS = \
	$O/assemble_MPI_scalar_gpu.o \
	$O/assemble_MPI_vector_gpu.o \
	$O/check_fields_gpu.o \
	$O/compute_add_sources_elastic_gpu.o \
	$O/compute_coupling_gpu.o \
	$O/compute_forces_crust_mantle_gpu.o \
	$O/compute_forces_inner_core_gpu.o \
	$O/compute_forces_outer_core_gpu.o \
	$O/compute_kernels_gpu.o \
	$O/compute_stacey_acoustic_gpu.o \
	$O/compute_stacey_elastic_gpu.o \
	$O/initialize_gpu.o \
	$O/noise_tomography_gpu.o \
	$O/prepare_mesh_constants_gpu.o \
	$O/transfer_fields_gpu.o \
	$O/update_displacement_gpu.o \
	$O/write_seismograms_gpu.o \
	$O/save_and_compare_cpu_vs_gpu.o \
	$(EMPTY_MACRO)

#######################################

ifeq ($(GPU_CUDA_AND_OCL),yes)
gpu_OBJECTS:=$(subst .o,.cuda-ocl.o,${gpu_OBJECTS})
endif

ifneq ($(GPU_CUDA_AND_OCL),yes)
ifeq ($(OCL), yes)
gpu_OBJECTS:=$(subst .o,.ocl.o,${gpu_OBJECTS})
endif
ifeq ($(CUDA),yes)
gpu_OBJECTS:=$(subst .o,.cuda.o,${gpu_OBJECTS})
endif
endif

## compilation directories

S := ${S_TOP}/src/gpu
$(gpu_OBJECTS): S = ${S_TOP}/src/gpu

BOAST_DIR_NAME := kernels.gen
BOAST_DIR := ${S}/${BOAST_DIR_NAME}

###
### variables
###

NVCC_ARCHFLAGS := -arch sm_20 
NVCC_CFLAGS := -x cu $(CUDA_FLAGS)  $(NVCC_ARCHFLAGS)

BUILD_VERSION_TXT := with
SELECTOR_CFLAG :=

ifeq ($(CUDA),yes)
BUILD_VERSION_TXT += Cuda
CUDA_LINK = -lcudart -lstdc++ 
SELECTOR_CFLAG += -DUSE_CUDA

ifeq ($(CUDA5),yes)
cuda_DEVICE_OBJ = $O/cuda_device_obj.o 	$(EMPTY_MACRO)
BUILD_VERSION_TXT += (v5)
endif
endif

ifeq ($(OCL), yes)
BUILD_VERSION_TXT += OpenCL
LDFLAGS += -lOpenCL
SELECTOR_CFLAG += -DUSE_OPENCL

ifeq ($(CUDA),yes)
CUDA_LINK += -lOpenCL
endif
endif

BUILD_VERSION_TXT += support

###
### building rules
###

boast_kernels :
	cd $S/boast ;\
	mkdir ../$(BOAST_DIR_NAME) -p ;\
	ruby kernels.rb --output-dir ../$(BOAST_DIR_NAME)

###
### compilation
###

ifeq ($(CUDA5),yes)

$(cuda_specfem3D_DEVICE_OBJ): $(cuda_OBJECTS)
	${NVCCLINK} -o $(cuda_specfem3D_DEVICE_OBJ) $(cuda_OBJECTS)

endif

$O/%.cuda-ocl.o: $O/%.cuda.o
	cd $O && cp $(shell basename $<) $(shell basename $@)

$O/%.ocl.o: $S/%.c ${SETUP}/config.h $S/mesh_constants_gpu.h $S/prepare_constants_gpu.h
	${CC} -c $< -o $@ -I${SETUP} -I$(BOAST_DIR) $(OCL_CPU_CFLAGS) $(SELECTOR_CFLAG) -DOCL_GPU_CFLAGS=$(OCL_GPU_CFLAGS)

$O/%.cuda.o: $S/%.c ${SETUP}/config.h $S/mesh_constants_gpu.h $S/prepare_constants_gpu.h
	$(NVCC) -c $< -o $@ $(NVCC_CFLAGS) -I${SETUP} -I$(BOAST_DIR) $(SELECTOR_CFLAG)

print-%:
	@echo '$*=$($*)'
