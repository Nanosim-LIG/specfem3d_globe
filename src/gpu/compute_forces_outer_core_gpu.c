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

void outer_core (int nb_blocks_to_compute, Mesh *mp,
                 int iphase,
                 gpu_int_mem d_ibool,
                 gpu_realw_mem d_xix,
                 gpu_realw_mem d_xiy,
                 gpu_realw_mem d_xiz,
                 gpu_realw_mem d_etax,
                 gpu_realw_mem d_etay,
                 gpu_realw_mem d_etaz,
                 gpu_realw_mem d_gammax,
                 gpu_realw_mem d_gammay,
                 gpu_realw_mem d_gammaz,
                 realw time,
                 gpu_realw_mem d_A_array_rotation,
                 gpu_realw_mem d_B_array_rotation,
                 gpu_realw_mem d_b_A_array_rotation,
                 gpu_realw_mem d_b_B_array_rotation,
                 int FORWARD_OR_ADJOINT) {

#ifdef ENABLE_VERY_SLOW_ERROR_CHECKING
  exit_on_gpu_error ("before outer_core kernel Kernel_2");
#endif

  // if the grid can handle the number of blocks, we let it be 1D
  // grid_2_x = nb_elem_color;
  // nb_elem_color is just how many blocks we are computing now
  int blocksize = NGLL3_PADDED;

  int num_blocks_x, num_blocks_y;
  get_blocks_xy (nb_blocks_to_compute, &num_blocks_x, &num_blocks_y);

#ifdef USE_OPENCL
  if (run_opencl) {
    if (FORWARD_OR_ADJOINT != 1 && FORWARD_OR_ADJOINT != 3) {
      goto skipexec;
    } else if (FORWARD_OR_ADJOINT == 3) {
      // debug
      DEBUG_BACKWARD_FORCES ();
    }

    size_t global_work_size[2];
    size_t local_work_size[2];

    clCheck (clSetKernelArg (mocl.kernels.outer_core_impl_kernel, 0, sizeof (int), (void *) &nb_blocks_to_compute));
    clCheck (clSetKernelArg (mocl.kernels.outer_core_impl_kernel, 1, sizeof (int), (void *) &mp->NGLOB_OUTER_CORE));
    clCheck (clSetKernelArg (mocl.kernels.outer_core_impl_kernel, 2, sizeof (gpu_int_mem), (void *) &d_ibool));
    clCheck (clSetKernelArg (mocl.kernels.outer_core_impl_kernel, 3, sizeof (gpu_int_mem), (void *) &mp->d_phase_ispec_inner_outer_core));
    clCheck (clSetKernelArg (mocl.kernels.outer_core_impl_kernel, 4, sizeof (int), (void *) &mp->num_phase_ispec_outer_core));
    clCheck (clSetKernelArg (mocl.kernels.outer_core_impl_kernel, 5, sizeof (int), (void *) &iphase));
    clCheck (clSetKernelArg (mocl.kernels.outer_core_impl_kernel, 6, sizeof (int), (void *) &mp->use_mesh_coloring_gpu));

    if (FORWARD_OR_ADJOINT == 1) {
      clCheck (clSetKernelArg (mocl.kernels.outer_core_impl_kernel, 7, sizeof (gpu_realw_mem), (void *) &mp->d_displ_outer_core));
      clCheck (clSetKernelArg (mocl.kernels.outer_core_impl_kernel, 8, sizeof (gpu_realw_mem), (void *) &mp->d_accel_outer_core));

    } else {
      clCheck (clSetKernelArg (mocl.kernels.outer_core_impl_kernel, 7, sizeof (gpu_realw_mem), (void *) &mp->d_b_displ_outer_core));
      clCheck (clSetKernelArg (mocl.kernels.outer_core_impl_kernel, 8, sizeof (gpu_realw_mem), (void *) &mp->d_b_accel_outer_core));
    }

    clCheck (clSetKernelArg (mocl.kernels.outer_core_impl_kernel, 9, sizeof (gpu_realw_mem), (void *) &d_xix));
    clCheck (clSetKernelArg (mocl.kernels.outer_core_impl_kernel, 10, sizeof (gpu_realw_mem), (void *) &d_xiy));
    clCheck (clSetKernelArg (mocl.kernels.outer_core_impl_kernel, 11, sizeof (gpu_realw_mem), (void *) &d_xiz));
    clCheck (clSetKernelArg (mocl.kernels.outer_core_impl_kernel, 12, sizeof (gpu_realw_mem), (void *) &d_etax));
    clCheck (clSetKernelArg (mocl.kernels.outer_core_impl_kernel, 13, sizeof (gpu_realw_mem), (void *) &d_etay));
    clCheck (clSetKernelArg (mocl.kernels.outer_core_impl_kernel, 14, sizeof (gpu_realw_mem), (void *) &d_etaz));
    clCheck (clSetKernelArg (mocl.kernels.outer_core_impl_kernel, 15, sizeof (gpu_realw_mem), (void *) &d_gammax));
    clCheck (clSetKernelArg (mocl.kernels.outer_core_impl_kernel, 16, sizeof (gpu_realw_mem), (void *) &d_gammay));
    clCheck (clSetKernelArg (mocl.kernels.outer_core_impl_kernel, 17, sizeof (gpu_realw_mem), (void *) &d_gammaz));
    clCheck (clSetKernelArg (mocl.kernels.outer_core_impl_kernel, 18, sizeof (gpu_realw_mem), (void *) &mp->d_hprime_xx));
    clCheck (clSetKernelArg (mocl.kernels.outer_core_impl_kernel, 19, sizeof (gpu_realw_mem), (void *) &mp->d_hprimewgll_xx));
    clCheck (clSetKernelArg (mocl.kernels.outer_core_impl_kernel, 20, sizeof (gpu_realw_mem), (void *) &mp->d_wgllwgll_xy));
    clCheck (clSetKernelArg (mocl.kernels.outer_core_impl_kernel, 21, sizeof (gpu_realw_mem), (void *) &mp->d_wgllwgll_xz));
    clCheck (clSetKernelArg (mocl.kernels.outer_core_impl_kernel, 22, sizeof (gpu_realw_mem), (void *) &mp->d_wgllwgll_yz));
    clCheck (clSetKernelArg (mocl.kernels.outer_core_impl_kernel, 23, sizeof (int), (void *) &mp->gravity));
    clCheck (clSetKernelArg (mocl.kernels.outer_core_impl_kernel, 24, sizeof (gpu_realw_mem), (void *) &mp->d_xstore_outer_core));
    clCheck (clSetKernelArg (mocl.kernels.outer_core_impl_kernel, 25, sizeof (gpu_realw_mem), (void *) &mp->d_ystore_outer_core));
    clCheck (clSetKernelArg (mocl.kernels.outer_core_impl_kernel, 26, sizeof (gpu_realw_mem), (void *) &mp->d_zstore_outer_core));
    clCheck (clSetKernelArg (mocl.kernels.outer_core_impl_kernel, 27, sizeof (gpu_realw_mem), (void *) &mp->d_d_ln_density_dr_table));
    clCheck (clSetKernelArg (mocl.kernels.outer_core_impl_kernel, 28, sizeof (gpu_realw_mem), (void *) &mp->d_minus_rho_g_over_kappa_fluid));
    clCheck (clSetKernelArg (mocl.kernels.outer_core_impl_kernel, 29, sizeof (gpu_realw_mem), (void *) &mp->d_wgll_cube));
    clCheck (clSetKernelArg (mocl.kernels.outer_core_impl_kernel, 30, sizeof (int), (void *) &mp->rotation));
    clCheck (clSetKernelArg (mocl.kernels.outer_core_impl_kernel, 31, sizeof (realw), (void *) &time));
    if (FORWARD_OR_ADJOINT == 1) {
      clCheck (clSetKernelArg (mocl.kernels.outer_core_impl_kernel, 32, sizeof (realw), (void *) &mp->two_omega_earth));
      clCheck (clSetKernelArg (mocl.kernels.outer_core_impl_kernel, 33, sizeof (realw), (void *) &mp->deltat));

      clCheck (clSetKernelArg (mocl.kernels.outer_core_impl_kernel, 34, sizeof (gpu_realw_mem), (void *) &d_A_array_rotation));
      clCheck (clSetKernelArg (mocl.kernels.outer_core_impl_kernel, 35, sizeof (gpu_realw_mem), (void *) &d_B_array_rotation));

    } else {
      clCheck (clSetKernelArg (mocl.kernels.outer_core_impl_kernel, 32, sizeof (realw), (void *) &mp->b_two_omega_earth));
      clCheck (clSetKernelArg (mocl.kernels.outer_core_impl_kernel, 33, sizeof (realw), (void *) &mp->b_deltat));

      clCheck (clSetKernelArg (mocl.kernels.outer_core_impl_kernel, 34, sizeof (gpu_realw_mem), (void *) &d_b_A_array_rotation));
      clCheck (clSetKernelArg (mocl.kernels.outer_core_impl_kernel, 35, sizeof (gpu_realw_mem), (void *) &d_b_B_array_rotation));
    }

    clCheck (clSetKernelArg (mocl.kernels.outer_core_impl_kernel, 36, sizeof (int), (void *) &mp->NSPEC_OUTER_CORE));

    clCheck (clSetKernelArg (mocl.kernels.outer_core_impl_kernel, 37, sizeof (cl_mem), (void *) &d_displ_oc_tex));
    clCheck (clSetKernelArg (mocl.kernels.outer_core_impl_kernel, 38, sizeof (cl_mem), (void *) &d_accel_oc_tex));
    clCheck (clSetKernelArg (mocl.kernels.outer_core_impl_kernel, 39, sizeof (cl_mem), (void *) &d_hprime_xx_oc_tex));

    local_work_size[0] = blocksize;
    local_work_size[1] = 1;
    global_work_size[0] = num_blocks_x * blocksize;
    global_work_size[1] = num_blocks_y;

    clCheck (clEnqueueNDRangeKernel (mocl.command_queue, mocl.kernels.outer_core_impl_kernel, 2, NULL,
                                     global_work_size, local_work_size, 0, NULL, NULL));

  }
skipexec:
#endif
#ifdef USE_CUDA
  if (run_cuda) {
    dim3 grid(num_blocks_x,num_blocks_y);
    dim3 threads(blocksize,1,1);

    if( FORWARD_OR_ADJOINT == 1 ){
      outer_core_impl_kernel<<<grid,threads>>>(nb_blocks_to_compute,
                                               mp->NGLOB_OUTER_CORE,
                                               d_ibool.cuda,
                                               mp->d_phase_ispec_inner_outer_core.cuda,
                                               mp->num_phase_ispec_outer_core,
                                               iphase,
                                               mp->use_mesh_coloring_gpu,
                                               mp->d_displ_outer_core.cuda,
                                               mp->d_accel_outer_core.cuda,
                                               d_xix.cuda, d_xiy.cuda, d_xiz.cuda,
                                               d_etax.cuda, d_etay.cuda, d_etaz.cuda,
                                               d_gammax.cuda, d_gammay.cuda, d_gammaz.cuda,
                                               mp->d_hprime_xx.cuda,
                                               mp->d_hprimewgll_xx.cuda,
                                               mp->d_wgllwgll_xy.cuda, mp->d_wgllwgll_xz.cuda, mp->d_wgllwgll_yz.cuda,
                                               mp->gravity,
                                               mp->d_xstore_outer_core.cuda,mp->d_ystore_outer_core.cuda,mp->d_zstore_outer_core.cuda,
                                               mp->d_d_ln_density_dr_table.cuda,
                                               mp->d_minus_rho_g_over_kappa_fluid.cuda,
                                               mp->d_wgll_cube.cuda,
                                               mp->rotation,
                                               time,
                                               mp->two_omega_earth,
                                               mp->deltat,
                                               d_A_array_rotation.cuda,
                                               d_B_array_rotation.cuda,
                                               mp->NSPEC_OUTER_CORE);
    }else if( FORWARD_OR_ADJOINT == 3 ){
      // debug
      DEBUG_BACKWARD_FORCES();

      outer_core_impl_kernel<<<grid,threads>>>(nb_blocks_to_compute,
                                               mp->NGLOB_OUTER_CORE,
                                               d_ibool.cuda,
                                               mp->d_phase_ispec_inner_outer_core.cuda,
                                               mp->num_phase_ispec_outer_core,
                                               iphase,
                                               mp->use_mesh_coloring_gpu,
                                               mp->d_b_displ_outer_core.cuda,
                                               mp->d_b_accel_outer_core.cuda,
                                               d_xix.cuda, d_xiy.cuda, d_xiz.cuda,
                                               d_etax.cuda, d_etay.cuda, d_etaz.cuda,
                                               d_gammax.cuda, d_gammay.cuda, d_gammaz.cuda,
                                               mp->d_hprime_xx.cuda,
                                               mp->d_hprimewgll_xx.cuda,
                                               mp->d_wgllwgll_xy.cuda, mp->d_wgllwgll_xz.cuda, mp->d_wgllwgll_yz.cuda,
                                               mp->gravity,
                                               mp->d_xstore_outer_core.cuda,mp->d_ystore_outer_core.cuda,mp->d_zstore_outer_core.cuda,
                                               mp->d_d_ln_density_dr_table.cuda,
                                               mp->d_minus_rho_g_over_kappa_fluid.cuda,
                                               mp->d_wgll_cube.cuda,
                                               mp->rotation,
                                               time,
                                               mp->b_two_omega_earth,
                                               mp->b_deltat,
                                               d_b_A_array_rotation.cuda,
                                               d_b_B_array_rotation.cuda,
                                               mp->NSPEC_OUTER_CORE);
    }
  }
#endif
#ifdef ENABLE_VERY_SLOW_ERROR_CHECKING
  exit_on_gpu_error ("kernel outer_core");
#endif
}

/*----------------------------------------------------------------------------------------------- */

// main compute_forces_outer_core OCL routine

/*----------------------------------------------------------------------------------------------- */

extern EXTERN_LANG
void FC_FUNC_ (compute_forces_outer_core_gpu,
               COMPUTE_FORCES_OUTER_CORE_OCL) (long *Mesh_pointer_f,
                                               int *iphase,
                                               realw *time_f,
                                               int *FORWARD_OR_ADJOINT_f) {

  TRACE ("compute_forces_outer_core_ocl");

  Mesh *mp = (Mesh *) (*Mesh_pointer_f);   // get Mesh from fortran integer wrapper
  realw time = *time_f;
  int FORWARD_OR_ADJOINT = *FORWARD_OR_ADJOINT_f;
  int num_elements;

  if (*iphase == 1) {
    num_elements = mp->nspec_outer_outer_core;
  } else {
    num_elements = mp->nspec_inner_outer_core;
  }

  if (num_elements == 0)
    return;

  // mesh coloring
  if (mp->use_mesh_coloring_gpu) {

    // note: array offsets require sorted arrays, such that e.g. ibool starts with elastic elements
    //         and followed by acoustic ones.
    //         acoustic elements also start with outer than inner element ordering

    int nb_colors, nb_blocks_to_compute;
    int istart;
    int offset, offset_nonpadded;

    // sets up color loop
    if (mp->NSPEC_OUTER_CORE > COLORING_MIN_NSPEC_OUTER_CORE) {
      if (*iphase == 1) {
        // outer elements
        nb_colors = mp->num_colors_outer_outer_core;
        istart = 0;

        // array offsets
        offset = 0;
        offset_nonpadded = 0;
      } else {
        // inner element colors (start after outer elements)
        nb_colors = mp->num_colors_outer_outer_core + mp->num_colors_inner_outer_core;
        istart = mp->num_colors_outer_outer_core;

        // array offsets (inner elements start after outer ones)
        offset = mp->nspec_outer_outer_core * NGLL3_PADDED;
        offset_nonpadded = mp->nspec_outer_outer_core * NGLL3;
      }
    } else {

      // poor element count, only use 1 color per inner/outer run

      if (*iphase == 1) {
        // outer elements
        nb_colors = 1;
        istart = 0;

        // array offsets
        offset = 0;
        offset_nonpadded = 0;
      } else {
        // inner element colors (start after outer elements)
        nb_colors = 1;
        istart = 0;

        // array offsets (inner elements start after outer ones)
        offset = mp->nspec_outer_outer_core * NGLL3_PADDED;
        offset_nonpadded = mp->nspec_outer_outer_core * NGLL3;
      }
    }

    // loops over colors
    int icolor;

    for (icolor = istart; icolor < nb_colors; icolor++) {

      // gets number of elements for this color
      if (mp->NSPEC_OUTER_CORE > COLORING_MIN_NSPEC_OUTER_CORE) {
        nb_blocks_to_compute = mp->h_num_elem_colors_outer_core[icolor];
      } else {
        nb_blocks_to_compute = num_elements;
      }

      INITIALIZE_OFFSET();

      //create cl_mem object _buffer + _ + _offset
      INIT_OFFSET(d_ibool_outer_core, offset_nonpadded);
      INIT_OFFSET(d_xix_outer_core, offset);
      INIT_OFFSET(d_xiy_outer_core, offset);
      INIT_OFFSET(d_xiz_outer_core, offset);
      INIT_OFFSET(d_etax_outer_core, offset);
      INIT_OFFSET(d_etay_outer_core, offset);
      INIT_OFFSET(d_etaz_outer_core, offset);
      INIT_OFFSET(d_gammax_outer_core, offset);
      INIT_OFFSET(d_gammay_outer_core, offset);
      INIT_OFFSET(d_gammaz_outer_core, offset);
      INIT_OFFSET(d_A_array_rotation, offset_nonpadded);
      INIT_OFFSET(d_B_array_rotation, offset_nonpadded);
      INIT_OFFSET(d_b_A_array_rotation, offset_nonpadded);
      INIT_OFFSET(d_b_B_array_rotation, offset_nonpadded);

      outer_core (nb_blocks_to_compute, mp,
                  *iphase,
                  PASS_OFFSET(d_ibool_outer_core, offset_nonpadded),
                  PASS_OFFSET(d_xix_outer_core, offset),
                  PASS_OFFSET(d_xiy_outer_core, offset),
                  PASS_OFFSET(d_xiz_outer_core, offset),
                  PASS_OFFSET(d_etax_outer_core, offset),
                  PASS_OFFSET(d_etay_outer_core, offset),
                  PASS_OFFSET(d_etaz_outer_core, offset),
                  PASS_OFFSET(d_gammax_outer_core, offset),
                  PASS_OFFSET(d_gammay_outer_core, offset),
                  PASS_OFFSET(d_gammaz_outer_core, offset),
                  time,
                  PASS_OFFSET(d_A_array_rotation, offset_nonpadded),
                  PASS_OFFSET(d_B_array_rotation, offset_nonpadded),
                  PASS_OFFSET(d_b_A_array_rotation, offset_nonpadded),
                  PASS_OFFSET(d_b_B_array_rotation, offset_nonpadded),
                  FORWARD_OR_ADJOINT);

      RELEASE_OFFSET(d_ibool_outer_core, offset_nonpadded);
      RELEASE_OFFSET(d_xix_outer_core, offset);
      RELEASE_OFFSET(d_xiy_outer_core, offset);
      RELEASE_OFFSET(d_xiz_outer_core, offset);
      RELEASE_OFFSET(d_etax_outer_core, offset);
      RELEASE_OFFSET(d_etay_outer_core, offset);
      RELEASE_OFFSET(d_etaz_outer_core, offset);
      RELEASE_OFFSET(d_gammax_outer_core, offset);
      RELEASE_OFFSET(d_gammay_outer_core, offset);
      RELEASE_OFFSET(d_gammaz_outer_core, offset);
      RELEASE_OFFSET(d_A_array_rotation, offset_nonpadded);
      RELEASE_OFFSET(d_B_array_rotation, offset_nonpadded);
      RELEASE_OFFSET(d_b_A_array_rotation, offset_nonpadded);
      RELEASE_OFFSET(d_b_B_array_rotation, offset_nonpadded);

      // for padded and aligned arrays
      offset += nb_blocks_to_compute * NGLL3_PADDED;
      // for no-aligned arrays
      offset_nonpadded += nb_blocks_to_compute * NGLL3;
    }

  } else {

    // no mesh coloring: uses atomic updates
    outer_core (num_elements, mp,
                *iphase,
                mp->d_ibool_outer_core,
                mp->d_xix_outer_core,
                mp->d_xiy_outer_core,
                mp->d_xiz_outer_core,
                mp->d_etax_outer_core,
                mp->d_etay_outer_core,
                mp->d_etaz_outer_core,
                mp->d_gammax_outer_core,
                mp->d_gammay_outer_core,
                mp->d_gammaz_outer_core,
                time,
                mp->d_A_array_rotation,
                mp->d_B_array_rotation,
                mp->d_b_A_array_rotation,
                mp->d_b_B_array_rotation,
                FORWARD_OR_ADJOINT);
  }

#ifdef ENABLE_VERY_SLOW_ERROR_CHECKING
  //double end_time = get_time ();
  //printf ("Elapsed time: %e\n", end_time-start_time);
  exit_on_gpu_error ("compute_forces_outer_core_ocl");
#endif
}
