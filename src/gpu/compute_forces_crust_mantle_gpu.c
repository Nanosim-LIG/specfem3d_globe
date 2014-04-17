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

#ifdef USE_CUDA
#ifdef USE_TEXTURES_FIELDS
//forward
realw_texture d_displ_cm_tex;
realw_texture d_accel_cm_tex;
//backward/reconstructed
realw_texture d_b_displ_cm_tex;
realw_texture d_b_accel_cm_tex;

//note: texture variables are implicitly static, and cannot be passed as arguments to cuda kernels;
//      thus, 1) we thus use if-statements (FORWARD_OR_ADJOINT) to determine from which texture to fetch from
//            2) we use templates
//      since if-statements are a bit slower as the variable is only known at runtime, we use option 2)

// templates definitions
template<int FORWARD_OR_ADJOINT> __device__ float texfetch_displ_cm(int x);
template<int FORWARD_OR_ADJOINT> __device__ float texfetch_accel_cm(int x);

// templates for texture fetching
// FORWARD_OR_ADJOINT == 1 <- forward arrays
template<> __device__ float texfetch_displ_cm<1>(int x) { return tex1Dfetch(d_displ_cm_tex, x); }
template<> __device__ float texfetch_accel_cm<1>(int x) { return tex1Dfetch(d_accel_cm_tex, x); }
// FORWARD_OR_ADJOINT == 3 <- backward/reconstructed arrays
template<> __device__ float texfetch_displ_cm<3>(int x) { return tex1Dfetch(d_b_displ_cm_tex, x); }
template<> __device__ float texfetch_accel_cm<3>(int x) { return tex1Dfetch(d_b_accel_cm_tex, x); }
#endif

#ifdef USE_TEXTURES_CONSTANTS
realw_texture d_hprime_xx_tex;
__constant__ size_t d_hprime_xx_tex_offset;
// weighted
realw_texture d_hprimewgll_xx_tex;
__constant__ size_t d_hprimewgll_xx_tex_offset;
#endif
#endif

template<int FORWARD_OR_ADJOINT> __global__ void
#ifdef USE_LAUNCH_BOUNDS
// adds compiler specification
__launch_bounds__(NGLL3_PADDED,LAUNCH_MIN_BLOCKS)
#endif
// main kernel
crust_mantle_impl_kernel_tmp( int nb_blocks_to_compute,
                            const int* d_ibool,
                            const int* d_ispec_is_tiso,
                            const int* d_phase_ispec_inner,
                            int num_phase_ispec,
                            const int d_iphase,
                            realw deltat,
                            const int use_mesh_coloring_gpu,
                            realw_const_p d_displ,
                            realw_p d_accel,
                            realw_const_p d_xix, realw_const_p d_xiy, realw_const_p d_xiz,
                            realw_const_p d_etax, realw_const_p d_etay, realw_const_p d_etaz,
                            realw_const_p d_gammax, realw_const_p d_gammay, realw_const_p d_gammaz,
                            realw_const_p d_hprime_xx,
                            realw_const_p d_hprimewgll_xx,
                            realw_const_p d_wgllwgll_xy,
                            realw_const_p d_wgllwgll_xz,
                            realw_const_p d_wgllwgll_yz,
                            realw_const_p d_kappavstore,
                            realw_const_p d_muvstore,
                            realw_const_p d_kappahstore,
                            realw_const_p d_muhstore,
                            realw_const_p d_eta_anisostore,
                            const int COMPUTE_AND_STORE_STRAIN,
                            realw_p epsilondev_xx,realw_p epsilondev_yy,realw_p epsilondev_xy,
                            realw_p epsilondev_xz,realw_p epsilondev_yz,
                            realw_p epsilon_trace_over_3,
                            const int ATTENUATION,
                            const int PARTIAL_PHYS_DISPERSION_ONLY,
                            const int USE_3D_ATTENUATION_ARRAYS,
                            realw_const_p one_minus_sum_beta,
                            realw_const_p factor_common,
                            realw_p R_xx, realw_p R_yy, realw_p R_xy, realw_p R_xz, realw_p R_yz,
                            realw_const_p alphaval,
                            realw_const_p betaval,
                            realw_const_p gammaval,
                            const int ANISOTROPY,
                            realw_const_p d_c11store,realw_const_p d_c12store,realw_const_p d_c13store,
                            realw_const_p d_c14store,realw_const_p d_c15store,realw_const_p d_c16store,
                            realw_const_p d_c22store,realw_const_p d_c23store,realw_const_p d_c24store,
                            realw_const_p d_c25store,realw_const_p d_c26store,realw_const_p d_c33store,
                            realw_const_p d_c34store,realw_const_p d_c35store,realw_const_p d_c36store,
                            realw_const_p d_c44store,realw_const_p d_c45store,realw_const_p d_c46store,
                            realw_const_p d_c55store,realw_const_p d_c56store,realw_const_p d_c66store,
                            const int GRAVITY,
                            realw_const_p d_xstore,realw_const_p d_ystore,realw_const_p d_zstore,
                            realw_const_p d_minus_gravity_table,
                            realw_const_p d_minus_deriv_gravity_table,
                            realw_const_p d_density_table,
                            realw_const_p wgll_cube,
                            const int NSPEC_CRUST_MANTLE_STRAIN_ONLY ){
}

void crust_mantle (int nb_blocks_to_compute, Mesh *mp,
                   int iphase,
                   gpu_int_mem d_ibool,
                   gpu_int_mem d_ispec_is_tiso,
                   gpu_realw_mem d_xix,
                   gpu_realw_mem d_xiy,
                   gpu_realw_mem d_xiz,
                   gpu_realw_mem d_etax,
                   gpu_realw_mem d_etay,
                   gpu_realw_mem d_etaz,
                   gpu_realw_mem d_gammax,
                   gpu_realw_mem d_gammay,
                   gpu_realw_mem d_gammaz,
                   gpu_realw_mem d_kappavstore,
                   gpu_realw_mem d_muvstore,
                   gpu_realw_mem d_kappahstore,
                   gpu_realw_mem d_muhstore,
                   gpu_realw_mem d_eta_anisostore,
                   gpu_realw_mem d_epsilondev_xx,
                   gpu_realw_mem d_epsilondev_yy,
                   gpu_realw_mem d_epsilondev_xy,
                   gpu_realw_mem d_epsilondev_xz,
                   gpu_realw_mem d_epsilondev_yz,
                   gpu_realw_mem d_epsilon_trace_over_3,
                   gpu_realw_mem d_one_minus_sum_beta,
                   gpu_realw_mem d_factor_common,
                   gpu_realw_mem d_R_xx,
                   gpu_realw_mem d_R_yy,
                   gpu_realw_mem d_R_xy,
                   gpu_realw_mem d_R_xz,
                   gpu_realw_mem d_R_yz,
                   gpu_realw_mem d_c11store,
                   gpu_realw_mem d_c12store,
                   gpu_realw_mem d_c13store,
                   gpu_realw_mem d_c14store,
                   gpu_realw_mem d_c15store,
                   gpu_realw_mem d_c16store,
                   gpu_realw_mem d_c22store,
                   gpu_realw_mem d_c23store,
                   gpu_realw_mem d_c24store,
                   gpu_realw_mem d_c25store,
                   gpu_realw_mem d_c26store,
                   gpu_realw_mem d_c33store,
                   gpu_realw_mem d_c34store,
                   gpu_realw_mem d_c35store,
                   gpu_realw_mem d_c36store,
                   gpu_realw_mem d_c44store,
                   gpu_realw_mem d_c45store,
                   gpu_realw_mem d_c46store,
                   gpu_realw_mem d_c55store,
                   gpu_realw_mem d_c56store,
                   gpu_realw_mem d_c66store,
                   gpu_realw_mem d_b_epsilondev_xx,
                   gpu_realw_mem d_b_epsilondev_yy,
                   gpu_realw_mem d_b_epsilondev_xy,
                   gpu_realw_mem d_b_epsilondev_xz,
                   gpu_realw_mem d_b_epsilondev_yz,
                   gpu_realw_mem d_b_epsilon_trace_over_3,
                   gpu_realw_mem d_b_R_xx,
                   gpu_realw_mem d_b_R_yy,
                   gpu_realw_mem d_b_R_xy,
                   gpu_realw_mem d_b_R_xz,
                   gpu_realw_mem d_b_R_yz,
                   int FORWARD_OR_ADJOINT) {

#ifdef ENABLE_VERY_SLOW_ERROR_CHECKING
  exit_on_gpu_error ("before kernel crust_mantle");
#endif

  // if the grid can handle the number of blocks, we let it be 1D
  // grid_2_x = nb_elem_color;
  // nb_elem_color is just how many blocks we are computing now

  int blocksize = NGLL3_PADDED;

  int num_blocks_x, num_blocks_y;
  get_blocks_xy (nb_blocks_to_compute, &num_blocks_x, &num_blocks_y);

#ifdef USE_OPENCL
  if (run_opencl) {
    size_t global_work_size[2];
    size_t local_work_size[2];

    if (FORWARD_OR_ADJOINT != 1 && FORWARD_OR_ADJOINT != 3) {
      goto skipexec;
    } else if (FORWARD_OR_ADJOINT == 3) {
      DEBUG_BACKWARD_FORCES ();
    }

    clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 0, sizeof (int), (void *) &nb_blocks_to_compute));
    clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 1, sizeof (int), (void *) &mp->NGLOB_CRUST_MANTLE));
    clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 2, sizeof (gpu_int_mem), (void *) &d_ibool.ocl));
    clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 3, sizeof (gpu_int_mem), (void *) &d_ispec_is_tiso.ocl));
    clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 4, sizeof (gpu_int_mem), (void *) &mp->d_phase_ispec_inner_crust_mantle.ocl));
    clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 5, sizeof (int), (void *) &mp->num_phase_ispec_crust_mantle));
    clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 6, sizeof (int), (void *) &iphase));
    if (FORWARD_OR_ADJOINT == 1) {
      clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 7, sizeof (realw), (void *) &mp->deltat));
    } else {
      clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 7, sizeof (realw), (void *) &mp->b_deltat));
    }
    clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 8, sizeof (int), (void *) &mp->use_mesh_coloring_gpu));

    if (FORWARD_OR_ADJOINT == 1) {
      clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 9, sizeof (gpu_realw_mem), (void *) &mp->d_displ_crust_mantle.ocl));
      clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 10, sizeof (gpu_realw_mem), (void *) &mp->d_veloc_crust_mantle.ocl));
      clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 11, sizeof (gpu_realw_mem), (void *) &mp->d_accel_crust_mantle.ocl));
    } else {
      clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 9, sizeof (gpu_realw_mem), (void *) &mp->d_b_displ_crust_mantle.ocl));
      clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 10, sizeof (gpu_realw_mem), (void *) &mp->d_b_veloc_crust_mantle.ocl));
      clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 11, sizeof (gpu_realw_mem), (void *) &mp->d_b_accel_crust_mantle.ocl));
    }

    clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 12, sizeof (gpu_realw_mem), (void *) &d_xix.ocl));
    clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 13, sizeof (gpu_realw_mem), (void *) &d_xiy.ocl));
    clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 14, sizeof (gpu_realw_mem), (void *) &d_xiz.ocl));
    clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 15, sizeof (gpu_realw_mem), (void *) &d_etax.ocl));
    clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 16, sizeof (gpu_realw_mem), (void *) &d_etay.ocl));
    clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 17, sizeof (gpu_realw_mem), (void *) &d_etaz.ocl));
    clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 18, sizeof (gpu_realw_mem), (void *) &d_gammax.ocl));
    clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 19, sizeof (gpu_realw_mem), (void *) &d_gammay.ocl));
    clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 20, sizeof (gpu_realw_mem), (void *) &d_gammaz.ocl));
    clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 21, sizeof (gpu_realw_mem), (void *) &mp->d_hprime_xx.ocl));
    clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 22, sizeof (gpu_realw_mem), (void *) &mp->d_hprimewgll_xx.ocl));
    clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 23, sizeof (gpu_realw_mem), (void *) &mp->d_wgllwgll_xy.ocl));
    clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 24, sizeof (gpu_realw_mem), (void *) &mp->d_wgllwgll_xz.ocl));
    clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 25, sizeof (gpu_realw_mem), (void *) &mp->d_wgllwgll_yz.ocl));
    clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 26, sizeof (gpu_realw_mem), (void *) &d_kappavstore.ocl));
    clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 27, sizeof (gpu_realw_mem), (void *) &d_muvstore.ocl));
    clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 28, sizeof (gpu_realw_mem), (void *) &d_kappahstore.ocl));
    clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 29, sizeof (gpu_realw_mem), (void *) &d_muhstore.ocl));
    clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 30, sizeof (gpu_realw_mem), (void *) &d_eta_anisostore.ocl));
    clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 31, sizeof (int), (void *) &mp->compute_and_store_strain));

    if (FORWARD_OR_ADJOINT == 1) {
      clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 32, sizeof (gpu_realw_mem), (void *) &d_epsilondev_xx.ocl));
      clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 33, sizeof (gpu_realw_mem), (void *) &d_epsilondev_yy.ocl));
      clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 34, sizeof (gpu_realw_mem), (void *) &d_epsilondev_xy.ocl));
      clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 35, sizeof (gpu_realw_mem), (void *) &d_epsilondev_xz.ocl));
      clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 36, sizeof (gpu_realw_mem), (void *) &d_epsilondev_yz.ocl));
      clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 37, sizeof (gpu_realw_mem), (void *) &d_epsilon_trace_over_3.ocl));
    } else {
      clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 32, sizeof (gpu_realw_mem), (void *) &d_b_epsilondev_xx.ocl));
      clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 33, sizeof (gpu_realw_mem), (void *) &d_b_epsilondev_yy.ocl));
      clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 34, sizeof (gpu_realw_mem), (void *) &d_b_epsilondev_xy.ocl));
      clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 35, sizeof (gpu_realw_mem), (void *) &d_b_epsilondev_xz.ocl));
      clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 36, sizeof (gpu_realw_mem), (void *) &d_b_epsilondev_yz.ocl));
      clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 37, sizeof (gpu_realw_mem), (void *) &d_b_epsilon_trace_over_3.ocl));
    }

    clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 38, sizeof (int), (void *) &mp->attenuation));
    clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 39, sizeof (int), (void *) &mp->partial_phys_dispersion_only));
    clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 40, sizeof (int), (void *) &mp->use_3d_attenuation_arrays));
    clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 41, sizeof (gpu_realw_mem), (void *) &d_one_minus_sum_beta.ocl));
    clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 42, sizeof (gpu_realw_mem), (void *) &d_factor_common.ocl));

    if (FORWARD_OR_ADJOINT == 1) {
      clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 43, sizeof (gpu_realw_mem), (void *) &d_R_xx.ocl));
      clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 44, sizeof (gpu_realw_mem), (void *) &d_R_yy.ocl));
      clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 45, sizeof (gpu_realw_mem), (void *) &d_R_xy.ocl));
      clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 46, sizeof (gpu_realw_mem), (void *) &d_R_xz.ocl));
      clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 47, sizeof (gpu_realw_mem), (void *) &d_R_yz.ocl));
      clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 48, sizeof (gpu_realw_mem), (void *) &mp->d_alphaval.ocl));
      clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 49, sizeof (gpu_realw_mem), (void *) &mp->d_betaval.ocl));
      clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 50, sizeof (gpu_realw_mem), (void *) &mp->d_gammaval.ocl));
    } else {
      clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 43, sizeof (gpu_realw_mem), (void *) &d_b_R_xx.ocl));
      clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 44, sizeof (gpu_realw_mem), (void *) &d_b_R_yy.ocl));
      clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 45, sizeof (gpu_realw_mem), (void *) &d_b_R_xy.ocl));
      clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 46, sizeof (gpu_realw_mem), (void *) &d_b_R_xz.ocl));
      clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 47, sizeof (gpu_realw_mem), (void *) &d_b_R_yz.ocl));
      clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 48, sizeof (gpu_realw_mem), (void *) &mp->d_b_alphaval.ocl));
      clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 49, sizeof (gpu_realw_mem), (void *) &mp->d_b_betaval.ocl));
      clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 50, sizeof (gpu_realw_mem), (void *) &mp->d_b_gammaval.ocl));
    }

    clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 51, sizeof (int), (void *) &mp->anisotropic_3D_mantle));
    clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 52, sizeof (gpu_realw_mem), (void *) &d_c11store.ocl));
    clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 53, sizeof (gpu_realw_mem), (void *) &d_c12store.ocl));
    clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 54, sizeof (gpu_realw_mem), (void *) &d_c13store.ocl));
    clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 55, sizeof (gpu_realw_mem), (void *) &d_c14store.ocl));
    clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 56, sizeof (gpu_realw_mem), (void *) &d_c15store.ocl));
    clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 57, sizeof (gpu_realw_mem), (void *) &d_c16store.ocl));
    clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 58, sizeof (gpu_realw_mem), (void *) &d_c22store.ocl));
    clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 59, sizeof (gpu_realw_mem), (void *) &d_c23store.ocl));
    clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 60, sizeof (gpu_realw_mem), (void *) &d_c24store.ocl));
    clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 61, sizeof (gpu_realw_mem), (void *) &d_c25store.ocl));
    clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 62, sizeof (gpu_realw_mem), (void *) &d_c26store.ocl));
    clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 63, sizeof (gpu_realw_mem), (void *) &d_c33store.ocl));
    clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 64, sizeof (gpu_realw_mem), (void *) &d_c34store.ocl));
    clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 65, sizeof (gpu_realw_mem), (void *) &d_c35store.ocl));
    clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 66, sizeof (gpu_realw_mem), (void *) &d_c36store.ocl));
    clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 67, sizeof (gpu_realw_mem), (void *) &d_c44store.ocl));
    clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 68, sizeof (gpu_realw_mem), (void *) &d_c45store.ocl));
    clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 69, sizeof (gpu_realw_mem), (void *) &d_c46store.ocl));
    clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 70, sizeof (gpu_realw_mem), (void *) &d_c55store.ocl));
    clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 71, sizeof (gpu_realw_mem), (void *) &d_c56store.ocl));
    clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 72, sizeof (gpu_realw_mem), (void *) &d_c66store.ocl));
    clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 73, sizeof (int), (void *) &mp->gravity));
    clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 74, sizeof (gpu_realw_mem), (void *) &mp->d_xstore_crust_mantle.ocl));
    clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 75, sizeof (gpu_realw_mem), (void *) &mp->d_ystore_crust_mantle.ocl));
    clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 76, sizeof (gpu_realw_mem), (void *) &mp->d_zstore_crust_mantle.ocl));
    clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 77, sizeof (gpu_realw_mem), (void *) &mp->d_minus_gravity_table.ocl));
    clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 78, sizeof (gpu_realw_mem), (void *) &mp->d_minus_deriv_gravity_table.ocl));
    clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 79, sizeof (gpu_realw_mem), (void *) &mp->d_density_table.ocl));
    clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 80, sizeof (gpu_realw_mem), (void *) &mp->d_wgll_cube.ocl));
    clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 81, sizeof (int), (void *) &mp->NSPEC_CRUST_MANTLE_STRAIN_ONLY));

    clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 82, sizeof (cl_mem), (void *) &d_displ_cm_tex));
    clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 83, sizeof (cl_mem), (void *) &d_accel_cm_tex));
    clCheck (clSetKernelArg (mocl.kernels.crust_mantle_impl_kernel, 84, sizeof (cl_mem), (void *) &d_hprime_xx_cm_tex));

    local_work_size[0] = blocksize;
    local_work_size[1] = 1;
    global_work_size[0] = num_blocks_x * blocksize;
    global_work_size[1] = num_blocks_y;

    clCheck (clEnqueueNDRangeKernel (mocl.command_queue, mocl.kernels.crust_mantle_impl_kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL));
  }
skipexec:
#endif
#ifdef USE_CUDA
  if (run_cuda) {
    dim3 grid(num_blocks_x,num_blocks_y);
    dim3 threads(blocksize,1,1);

    if (FORWARD_OR_ADJOINT == 1) {
      // forward wavefields -> FORWARD_OR_ADJOINT == 1
      crust_mantle_impl_kernel_tmp<1><<<grid,threads,0,mp->compute_stream>>>(nb_blocks_to_compute,
                                                                      d_ibool.cuda,
                                                                      d_ispec_is_tiso.cuda,
                                                                      mp->d_phase_ispec_inner_crust_mantle.cuda,
                                                                      mp->num_phase_ispec_crust_mantle,
                                                                      iphase,
                                                                      mp->deltat,
                                                                      mp->use_mesh_coloring_gpu,
                                                                      mp->d_displ_crust_mantle.cuda,
                                                                      mp->d_accel_crust_mantle.cuda,
                                                                      d_xix.cuda, d_xiy.cuda, d_xiz.cuda,
                                                                      d_etax.cuda, d_etay.cuda, d_etaz.cuda,
                                                                      d_gammax.cuda, d_gammay.cuda, d_gammaz.cuda,
                                                                      mp->d_hprime_xx.cuda,
                                                                      mp->d_hprimewgll_xx.cuda,
                                                                      mp->d_wgllwgll_xy.cuda, mp->d_wgllwgll_xz.cuda, mp->d_wgllwgll_yz.cuda,
                                                                      d_kappavstore.cuda, d_muvstore.cuda,
                                                                      d_kappahstore.cuda, d_muhstore.cuda,
                                                                      d_eta_anisostore.cuda,
                                                                      mp->compute_and_store_strain,
                                                                      d_epsilondev_xx.cuda,d_epsilondev_yy.cuda,d_epsilondev_xy.cuda,
                                                                      d_epsilondev_xz.cuda,d_epsilondev_yz.cuda,
                                                                      d_epsilon_trace_over_3.cuda,
                                                                      mp->attenuation,
                                                                      mp->partial_phys_dispersion_only,
                                                                      mp->use_3d_attenuation_arrays,
                                                                      d_one_minus_sum_beta.cuda,d_factor_common.cuda,
                                                                      d_R_xx.cuda,d_R_yy.cuda,d_R_xy.cuda,d_R_xz.cuda,d_R_yz.cuda,
                                                                      mp->d_alphaval.cuda,mp->d_betaval.cuda,mp->d_gammaval.cuda,
                                                                      mp->anisotropic_3D_mantle,
                                                                      d_c11store.cuda,d_c12store.cuda,d_c13store.cuda,
                                                                      d_c14store.cuda,d_c15store.cuda,d_c16store.cuda,
                                                                      d_c22store.cuda,d_c23store.cuda,d_c24store.cuda,
                                                                      d_c25store.cuda,d_c26store.cuda,d_c33store.cuda,
                                                                      d_c34store.cuda,d_c35store.cuda,d_c36store.cuda,
                                                                      d_c44store.cuda,d_c45store.cuda,d_c46store.cuda,
                                                                      d_c55store.cuda,d_c56store.cuda,d_c66store.cuda,
                                                                      mp->gravity,
                                                                      mp->d_xstore_crust_mantle.cuda,mp->d_ystore_crust_mantle.cuda,mp->d_zstore_crust_mantle.cuda,
                                                                      mp->d_minus_gravity_table.cuda,
                                                                      mp->d_minus_deriv_gravity_table.cuda,
                                                                      mp->d_density_table.cuda,
                                                                      mp->d_wgll_cube.cuda,
                                                                      mp->NSPEC_CRUST_MANTLE_STRAIN_ONLY);
    } else if (FORWARD_OR_ADJOINT == 3) {
      // backward/reconstructed wavefields -> FORWARD_OR_ADJOINT == 3
      // debug
      DEBUG_BACKWARD_FORCES();
      crust_mantle_impl_kernel_tmp<3><<< grid,threads,0,mp->compute_stream>>>(nb_blocks_to_compute,
                                                  d_ibool.cuda,
                                                  d_ispec_is_tiso.cuda,
                                                  mp->d_phase_ispec_inner_crust_mantle.cuda,
                                                  mp->num_phase_ispec_crust_mantle,
                                                  iphase,
                                                  mp->b_deltat,
                                                  mp->use_mesh_coloring_gpu,
                                                  mp->d_b_displ_crust_mantle.cuda,
                                                  mp->d_b_accel_crust_mantle.cuda,
                                                  d_xix.cuda, d_xiy.cuda, d_xiz.cuda,
                                                  d_etax.cuda, d_etay.cuda, d_etaz.cuda,
                                                  d_gammax.cuda, d_gammay.cuda, d_gammaz.cuda,
                                                  mp->d_hprime_xx.cuda,
                                                  mp->d_hprimewgll_xx.cuda,
                                                  mp->d_wgllwgll_xy.cuda, mp->d_wgllwgll_xz.cuda, mp->d_wgllwgll_yz.cuda,
                                                  d_kappavstore.cuda, d_muvstore.cuda,
                                                  d_kappahstore.cuda, d_muhstore.cuda,
                                                  d_eta_anisostore.cuda,
                                                  mp->compute_and_store_strain,
                                                  d_b_epsilondev_xx.cuda,d_b_epsilondev_yy.cuda,d_b_epsilondev_xy.cuda,
                                                  d_b_epsilondev_xz.cuda,d_b_epsilondev_yz.cuda,
                                                  d_b_epsilon_trace_over_3.cuda,
                                                  mp->attenuation,
                                                  mp->partial_phys_dispersion_only,
                                                  mp->use_3d_attenuation_arrays,
                                                  d_one_minus_sum_beta.cuda,d_factor_common.cuda,
                                                  d_b_R_xx.cuda,d_b_R_yy.cuda,d_b_R_xy.cuda,d_b_R_xz.cuda,d_b_R_yz.cuda,
                                                  mp->d_b_alphaval.cuda,mp->d_b_betaval.cuda,mp->d_b_gammaval.cuda,
                                                  mp->anisotropic_3D_mantle,
                                                  d_c11store.cuda,d_c12store.cuda,d_c13store.cuda,
                                                  d_c14store.cuda,d_c15store.cuda,d_c16store.cuda,
                                                  d_c22store.cuda,d_c23store.cuda,d_c24store.cuda,
                                                  d_c25store.cuda,d_c26store.cuda,d_c33store.cuda,
                                                  d_c34store.cuda,d_c35store.cuda,d_c36store.cuda,
                                                  d_c44store.cuda,d_c45store.cuda,d_c46store.cuda,
                                                  d_c55store.cuda,d_c56store.cuda,d_c66store.cuda,
                                                  mp->gravity,
                                                  mp->d_xstore_crust_mantle.cuda,mp->d_ystore_crust_mantle.cuda,mp->d_zstore_crust_mantle.cuda,
                                                  mp->d_minus_gravity_table.cuda,
                                                  mp->d_minus_deriv_gravity_table.cuda,
                                                  mp->d_density_table.cuda,
                                                  mp->d_wgll_cube.cuda,
                                                  mp->NSPEC_CRUST_MANTLE_STRAIN_ONLY);
    }
  }
#endif
#ifdef ENABLE_VERY_SLOW_ERROR_CHECKING
  exit_on_gpu_error ("crust_mantle");
#endif
}

/*----------------------------------------------------------------------------------------------- */


extern EXTERN_LANG
void FC_FUNC_ (compute_forces_crust_mantle_gpu,
               COMPUTE_FORCES_CRUST_MANTLE_GPU) (long *Mesh_pointer_f,
                                                 int *iphase,
                                                 int *FORWARD_OR_ADJOINT_f) {

  TRACE ("compute_forces_crust_mantle_gpu");

  // get Mesh from fortran integer wrapper
  Mesh *mp = (Mesh *) *Mesh_pointer_f;

  int FORWARD_OR_ADJOINT = *FORWARD_OR_ADJOINT_f;
  int num_elements;

  if (*iphase == 1) {
    num_elements = mp->nspec_outer_crust_mantle;
  } else {
    num_elements = mp->nspec_inner_crust_mantle;
  }

  // checks if anything to do
  if (num_elements == 0)
    return;

  // mesh coloring
  if (mp->use_mesh_coloring_gpu) {

    // note: array offsets require sorted arrays, such that e.g. ibool starts with elastic elements
    //         and followed by acoustic ones.
    //         elastic elements also start with outer than inner element ordering

    int nb_colors, nb_blocks_to_compute;
    int istart;
    int offset, offset_nonpadded;
    int offset_nonpadded_att1, offset_nonpadded_att2, offset_nonpadded_att3;
    int offset_nonpadded_strain;
    int offset_ispec;

    // sets up color loop
    if (*iphase == 1) {
      // outer elements
      nb_colors = mp->num_colors_outer_crust_mantle;
      istart = 0;

      // array offsets
      offset = 0;
      offset_nonpadded = 0;
      offset_nonpadded_att1 = 0;
      offset_nonpadded_att2 = 0;
      offset_nonpadded_att3 = 0;
      offset_nonpadded_strain = 0;
      offset_ispec = 0;
    } else {
      // inner elements (start after outer elements)
      nb_colors = mp->num_colors_outer_crust_mantle + mp->num_colors_inner_crust_mantle;
      istart = mp->num_colors_outer_crust_mantle;

      // array offsets
      offset = mp->nspec_outer_crust_mantle * NGLL3_PADDED;
      offset_nonpadded = mp->nspec_outer_crust_mantle * NGLL3;
      offset_nonpadded_att1 = mp->nspec_outer_crust_mantle * NGLL3 * N_SLS;

      // for factor_common array
      if (mp->use_3d_attenuation_arrays) {
        offset_nonpadded_att2 = mp->nspec_outer_crust_mantle * NGLL3;
        offset_nonpadded_att3 = mp->nspec_outer_crust_mantle * NGLL3 * N_SLS;
      } else {
        offset_nonpadded_att2 = mp->nspec_outer_crust_mantle * 1;
        offset_nonpadded_att3 = mp->nspec_outer_crust_mantle * 1 * N_SLS;
      }
      // for tiso models
      if (!mp->anisotropic_3D_mantle) {
        offset_ispec = mp->nspec_outer_crust_mantle;
      }
      // for strain
      if (mp->NSPEC_CRUST_MANTLE_STRAIN_ONLY != 1) {
        offset_nonpadded_strain = mp->nspec_outer_crust_mantle * NGLL3;
      }
    }

    // loops over colors
    int icolor;
    for (icolor = istart; icolor < nb_colors; icolor++) {
      nb_blocks_to_compute = mp->h_num_elem_colors_crust_mantle[icolor];

#ifdef ENABLE_VERY_SLOW_ERROR_CHECKING
      // checks
      if (nb_blocks_to_compute <= 0) {
        printf ("error number of color blocks in crust_mantle: %d -- color = %d \n",
                nb_blocks_to_compute, icolor);
        exit (EXIT_FAILURE);
      }
#endif
      INITIALIZE_OFFSET();

      INIT_OFFSET(d_ibool_crust_mantle, offset_nonpadded);
      INIT_OFFSET(d_ispec_is_tiso_crust_mantle, offset_ispec);
      INIT_OFFSET(d_xix_crust_mantle, offset);
      INIT_OFFSET(d_xiy_crust_mantle, offset);
      INIT_OFFSET(d_xiz_crust_mantle, offset);
      INIT_OFFSET(d_etax_crust_mantle, offset);
      INIT_OFFSET(d_etay_crust_mantle, offset);
      INIT_OFFSET(d_etaz_crust_mantle, offset);
      INIT_OFFSET(d_gammax_crust_mantle, offset);
      INIT_OFFSET(d_gammay_crust_mantle, offset);
      INIT_OFFSET(d_gammaz_crust_mantle, offset);
      INIT_OFFSET(d_kappavstore_crust_mantle, offset);
      INIT_OFFSET(d_muvstore_crust_mantle, offset);
      INIT_OFFSET(d_kappahstore_crust_mantle, offset);
      INIT_OFFSET(d_muhstore_crust_mantle, offset);
      INIT_OFFSET(d_eta_anisostore_crust_mantle, offset);
      INIT_OFFSET(d_epsilondev_xx_crust_mantle, offset_nonpadded);
      INIT_OFFSET(d_epsilondev_yy_crust_mantle, offset_nonpadded);
      INIT_OFFSET(d_epsilondev_xy_crust_mantle, offset_nonpadded);
      INIT_OFFSET(d_epsilondev_xz_crust_mantle, offset_nonpadded);
      INIT_OFFSET(d_epsilondev_yz_crust_mantle, offset_nonpadded);
      INIT_OFFSET(d_eps_trace_over_3_crust_mantle, offset_nonpadded_strain);
      INIT_OFFSET(d_one_minus_sum_beta_crust_mantle, offset_nonpadded_att2);
      INIT_OFFSET(d_factor_common_crust_mantle, offset_nonpadded_att3);
      INIT_OFFSET(d_R_xx_crust_mantle, offset_nonpadded_att1);
      INIT_OFFSET(d_R_yy_crust_mantle, offset_nonpadded_att1);
      INIT_OFFSET(d_R_xy_crust_mantle, offset_nonpadded_att1);
      INIT_OFFSET(d_R_xz_crust_mantle, offset_nonpadded_att1);
      INIT_OFFSET(d_R_yz_crust_mantle, offset_nonpadded_att1);
      INIT_OFFSET(d_c11store_crust_mantle, offset);
      INIT_OFFSET(d_c12store_crust_mantle, offset);
      INIT_OFFSET(d_c13store_crust_mantle, offset);
      INIT_OFFSET(d_c14store_crust_mantle, offset);
      INIT_OFFSET(d_c15store_crust_mantle, offset);
      INIT_OFFSET(d_c16store_crust_mantle, offset);
      INIT_OFFSET(d_c22store_crust_mantle, offset);
      INIT_OFFSET(d_c23store_crust_mantle, offset);
      INIT_OFFSET(d_c24store_crust_mantle, offset);
      INIT_OFFSET(d_c25store_crust_mantle, offset);
      INIT_OFFSET(d_c26store_crust_mantle, offset);
      INIT_OFFSET(d_c33store_crust_mantle, offset);
      INIT_OFFSET(d_c34store_crust_mantle, offset);
      INIT_OFFSET(d_c35store_crust_mantle, offset);
      INIT_OFFSET(d_c36store_crust_mantle, offset);
      INIT_OFFSET(d_c44store_crust_mantle, offset);
      INIT_OFFSET(d_c45store_crust_mantle, offset);
      INIT_OFFSET(d_c46store_crust_mantle, offset);
      INIT_OFFSET(d_c55store_crust_mantle, offset);
      INIT_OFFSET(d_c56store_crust_mantle, offset);
      INIT_OFFSET(d_c66store_crust_mantle, offset);
      INIT_OFFSET(d_b_epsilondev_xx_crust_mantle, offset_nonpadded);
      INIT_OFFSET(d_b_epsilondev_yy_crust_mantle, offset_nonpadded);
      INIT_OFFSET(d_b_epsilondev_xy_crust_mantle, offset_nonpadded);
      INIT_OFFSET(d_b_epsilondev_xz_crust_mantle, offset_nonpadded);
      INIT_OFFSET(d_b_epsilondev_yz_crust_mantle, offset_nonpadded);
      INIT_OFFSET(d_b_eps_trace_over_3_crust_mantle, offset_nonpadded);
      INIT_OFFSET(d_b_R_xx_crust_mantle, offset_nonpadded_att1);
      INIT_OFFSET(d_b_R_yy_crust_mantle, offset_nonpadded_att1);
      INIT_OFFSET(d_b_R_xy_crust_mantle, offset_nonpadded_att1);
      INIT_OFFSET(d_b_R_xz_crust_mantle, offset_nonpadded_att1);
      INIT_OFFSET(d_b_R_yz_crust_mantle, offset_nonpadded_att1);

      crust_mantle(nb_blocks_to_compute,mp,
                   *iphase,
                   PASS_OFFSET(d_ibool_crust_mantle, offset_nonpadded),
                   PASS_OFFSET(d_ispec_is_tiso_crust_mantle, offset_ispec),
                   PASS_OFFSET(d_xix_crust_mantle, offset),
                   PASS_OFFSET(d_xiy_crust_mantle, offset),
                   PASS_OFFSET(d_xiz_crust_mantle, offset),
                   PASS_OFFSET(d_etax_crust_mantle, offset),
                   PASS_OFFSET(d_etay_crust_mantle, offset),
                   PASS_OFFSET(d_etaz_crust_mantle, offset),
                   PASS_OFFSET(d_gammax_crust_mantle, offset),
                   PASS_OFFSET(d_gammay_crust_mantle, offset),
                   PASS_OFFSET(d_gammaz_crust_mantle, offset),
                   PASS_OFFSET(d_kappavstore_crust_mantle, offset),
                   PASS_OFFSET(d_muvstore_crust_mantle, offset),
                   PASS_OFFSET(d_kappahstore_crust_mantle, offset),
                   PASS_OFFSET(d_muhstore_crust_mantle, offset),
                   PASS_OFFSET(d_eta_anisostore_crust_mantle, offset),
                   PASS_OFFSET(d_epsilondev_xx_crust_mantle, offset_nonpadded),
                   PASS_OFFSET(d_epsilondev_yy_crust_mantle, offset_nonpadded),
                   PASS_OFFSET(d_epsilondev_xy_crust_mantle, offset_nonpadded),
                   PASS_OFFSET(d_epsilondev_xz_crust_mantle, offset_nonpadded),
                   PASS_OFFSET(d_epsilondev_yz_crust_mantle, offset_nonpadded),
                   PASS_OFFSET(d_eps_trace_over_3_crust_mantle, offset_nonpadded_strain),
                   PASS_OFFSET(d_one_minus_sum_beta_crust_mantle, offset_nonpadded_att2),
                   PASS_OFFSET(d_factor_common_crust_mantle, offset_nonpadded_att3),
                   PASS_OFFSET(d_R_xx_crust_mantle, offset_nonpadded_att1),
                   PASS_OFFSET(d_R_yy_crust_mantle, offset_nonpadded_att1),
                   PASS_OFFSET(d_R_xy_crust_mantle, offset_nonpadded_att1),
                   PASS_OFFSET(d_R_xz_crust_mantle, offset_nonpadded_att1),
                   PASS_OFFSET(d_R_yz_crust_mantle, offset_nonpadded_att1),
                   PASS_OFFSET(d_c11store_crust_mantle, offset),
                   PASS_OFFSET(d_c12store_crust_mantle, offset),
                   PASS_OFFSET(d_c13store_crust_mantle, offset),
                   PASS_OFFSET(d_c14store_crust_mantle, offset),
                   PASS_OFFSET(d_c15store_crust_mantle, offset),
                   PASS_OFFSET(d_c16store_crust_mantle, offset),
                   PASS_OFFSET(d_c22store_crust_mantle, offset),
                   PASS_OFFSET(d_c23store_crust_mantle, offset),
                   PASS_OFFSET(d_c24store_crust_mantle, offset),
                   PASS_OFFSET(d_c25store_crust_mantle, offset),
                   PASS_OFFSET(d_c26store_crust_mantle, offset),
                   PASS_OFFSET(d_c33store_crust_mantle, offset),
                   PASS_OFFSET(d_c34store_crust_mantle, offset),
                   PASS_OFFSET(d_c35store_crust_mantle, offset),
                   PASS_OFFSET(d_c36store_crust_mantle, offset),
                   PASS_OFFSET(d_c44store_crust_mantle, offset),
                   PASS_OFFSET(d_c45store_crust_mantle, offset),
                   PASS_OFFSET(d_c46store_crust_mantle, offset),
                   PASS_OFFSET(d_c55store_crust_mantle, offset),
                   PASS_OFFSET(d_c56store_crust_mantle, offset),
                   PASS_OFFSET(d_c66store_crust_mantle, offset),
                   PASS_OFFSET(d_b_epsilondev_xx_crust_mantle, offset_nonpadded),
                   PASS_OFFSET(d_b_epsilondev_yy_crust_mantle, offset_nonpadded),
                   PASS_OFFSET(d_b_epsilondev_xy_crust_mantle, offset_nonpadded),
                   PASS_OFFSET(d_b_epsilondev_xz_crust_mantle, offset_nonpadded),
                   PASS_OFFSET(d_b_epsilondev_yz_crust_mantle, offset_nonpadded),
                   PASS_OFFSET(d_b_eps_trace_over_3_crust_mantle, offset_nonpadded),
                   PASS_OFFSET(d_b_R_xx_crust_mantle, offset_nonpadded_att1),
                   PASS_OFFSET(d_b_R_yy_crust_mantle, offset_nonpadded_att1),
                   PASS_OFFSET(d_b_R_xy_crust_mantle, offset_nonpadded_att1),
                   PASS_OFFSET(d_b_R_xz_crust_mantle, offset_nonpadded_att1),
                   PASS_OFFSET(d_b_R_yz_crust_mantle, offset_nonpadded_att1),
                   FORWARD_OR_ADJOINT);

      RELEASE_OFFSET(d_ibool_crust_mantle, offset_nonpadded);
      RELEASE_OFFSET(d_ispec_is_tiso_crust_mantle, offset_ispec);
      RELEASE_OFFSET(d_xix_crust_mantle, offset);
      RELEASE_OFFSET(d_xiy_crust_mantle, offset);
      RELEASE_OFFSET(d_xiz_crust_mantle, offset);
      RELEASE_OFFSET(d_etax_crust_mantle, offset);
      RELEASE_OFFSET(d_etay_crust_mantle, offset);
      RELEASE_OFFSET(d_etaz_crust_mantle, offset);
      RELEASE_OFFSET(d_gammax_crust_mantle, offset);
      RELEASE_OFFSET(d_gammay_crust_mantle, offset);
      RELEASE_OFFSET(d_gammaz_crust_mantle, offset);
      RELEASE_OFFSET(d_kappavstore_crust_mantle, offset);
      RELEASE_OFFSET(d_muvstore_crust_mantle, offset);
      RELEASE_OFFSET(d_kappahstore_crust_mantle, offset);
      RELEASE_OFFSET(d_muhstore_crust_mantle, offset);
      RELEASE_OFFSET(d_eta_anisostore_crust_mantle, offset);
      RELEASE_OFFSET(d_epsilondev_xx_crust_mantle, offset_nonpadded);
      RELEASE_OFFSET(d_epsilondev_yy_crust_mantle, offset_nonpadded);
      RELEASE_OFFSET(d_epsilondev_xy_crust_mantle, offset_nonpadded);
      RELEASE_OFFSET(d_epsilondev_xz_crust_mantle, offset_nonpadded);
      RELEASE_OFFSET(d_epsilondev_yz_crust_mantle, offset_nonpadded);
      RELEASE_OFFSET(d_eps_trace_over_3_crust_mantle, offset_nonpadded_strain);
      RELEASE_OFFSET(d_one_minus_sum_beta_crust_mantle, offset_nonpadded_att2);
      RELEASE_OFFSET(d_factor_common_crust_mantle, offset_nonpadded_att3);
      RELEASE_OFFSET(d_R_xx_crust_mantle, offset_nonpadded_att1);
      RELEASE_OFFSET(d_R_yy_crust_mantle, offset_nonpadded_att1);
      RELEASE_OFFSET(d_R_xy_crust_mantle, offset_nonpadded_att1);
      RELEASE_OFFSET(d_R_xz_crust_mantle, offset_nonpadded_att1);
      RELEASE_OFFSET(d_R_yz_crust_mantle, offset_nonpadded_att1);
      RELEASE_OFFSET(d_c11store_crust_mantle, offset);
      RELEASE_OFFSET(d_c12store_crust_mantle, offset);
      RELEASE_OFFSET(d_c13store_crust_mantle, offset);
      RELEASE_OFFSET(d_c14store_crust_mantle, offset);
      RELEASE_OFFSET(d_c15store_crust_mantle, offset);
      RELEASE_OFFSET(d_c16store_crust_mantle, offset);
      RELEASE_OFFSET(d_c22store_crust_mantle, offset);
      RELEASE_OFFSET(d_c23store_crust_mantle, offset);
      RELEASE_OFFSET(d_c24store_crust_mantle, offset);
      RELEASE_OFFSET(d_c25store_crust_mantle, offset);
      RELEASE_OFFSET(d_c26store_crust_mantle, offset);
      RELEASE_OFFSET(d_c33store_crust_mantle, offset);
      RELEASE_OFFSET(d_c34store_crust_mantle, offset);
      RELEASE_OFFSET(d_c35store_crust_mantle, offset);
      RELEASE_OFFSET(d_c36store_crust_mantle, offset);
      RELEASE_OFFSET(d_c44store_crust_mantle, offset);
      RELEASE_OFFSET(d_c45store_crust_mantle, offset);
      RELEASE_OFFSET(d_c46store_crust_mantle, offset);
      RELEASE_OFFSET(d_c55store_crust_mantle, offset);
      RELEASE_OFFSET(d_c56store_crust_mantle, offset);
      RELEASE_OFFSET(d_c66store_crust_mantle, offset);
      RELEASE_OFFSET(d_b_epsilondev_xx_crust_mantle, offset_nonpadded);
      RELEASE_OFFSET(d_b_epsilondev_yy_crust_mantle, offset_nonpadded);
      RELEASE_OFFSET(d_b_epsilondev_xy_crust_mantle, offset_nonpadded);
      RELEASE_OFFSET(d_b_epsilondev_xz_crust_mantle, offset_nonpadded);
      RELEASE_OFFSET(d_b_epsilondev_yz_crust_mantle, offset_nonpadded);
      RELEASE_OFFSET(d_b_eps_trace_over_3_crust_mantle, offset_nonpadded);
      RELEASE_OFFSET(d_b_R_xx_crust_mantle, offset_nonpadded_att1);
      RELEASE_OFFSET(d_b_R_yy_crust_mantle, offset_nonpadded_att1);
      RELEASE_OFFSET(d_b_R_xy_crust_mantle, offset_nonpadded_att1);
      RELEASE_OFFSET(d_b_R_xz_crust_mantle, offset_nonpadded_att1);
      RELEASE_OFFSET(d_b_R_yz_crust_mantle, offset_nonpadded_att1);

      // for padded and aligned arrays
      offset += nb_blocks_to_compute * NGLL3_PADDED;
      // for no-aligned arrays
      offset_nonpadded += nb_blocks_to_compute * NGLL3;
      offset_nonpadded_att1 += nb_blocks_to_compute * NGLL3 * N_SLS;
      // for factor_common array
      if (mp->use_3d_attenuation_arrays) {
        offset_nonpadded_att2 += nb_blocks_to_compute * NGLL3;
        offset_nonpadded_att3 += nb_blocks_to_compute * NGLL3 * N_SLS;
      } else {
        offset_nonpadded_att2 += nb_blocks_to_compute * 1;
        offset_nonpadded_att3 += nb_blocks_to_compute * 1 * N_SLS;
      }
      // for tiso models
      if (!mp->anisotropic_3D_mantle) {
        offset_ispec += nb_blocks_to_compute;
      }
      // for strain
      if (mp->NSPEC_CRUST_MANTLE_STRAIN_ONLY != 1) {
        offset_nonpadded_strain += nb_blocks_to_compute * NGLL3;
      }

    }
    // icolor
  } else {

    // no mesh coloring: uses atomic updates
    crust_mantle(num_elements,mp,
                 *iphase,
                 mp->d_ibool_crust_mantle,
                 mp->d_ispec_is_tiso_crust_mantle,
                 mp->d_xix_crust_mantle,
                 mp->d_xiy_crust_mantle,
                 mp->d_xiz_crust_mantle,
                 mp->d_etax_crust_mantle,
                 mp->d_etay_crust_mantle,
                 mp->d_etaz_crust_mantle,
                 mp->d_gammax_crust_mantle,
                 mp->d_gammay_crust_mantle,
                 mp->d_gammaz_crust_mantle,
                 mp->d_kappavstore_crust_mantle,
                 mp->d_muvstore_crust_mantle,
                 mp->d_kappahstore_crust_mantle,
                 mp->d_muhstore_crust_mantle,
                 mp->d_eta_anisostore_crust_mantle,
                 mp->d_epsilondev_xx_crust_mantle,
                 mp->d_epsilondev_yy_crust_mantle,
                 mp->d_epsilondev_xy_crust_mantle,
                 mp->d_epsilondev_xz_crust_mantle,
                 mp->d_epsilondev_yz_crust_mantle,
                 mp->d_eps_trace_over_3_crust_mantle,
                 mp->d_one_minus_sum_beta_crust_mantle,
                 mp->d_factor_common_crust_mantle,
                 mp->d_R_xx_crust_mantle,
                 mp->d_R_yy_crust_mantle,
                 mp->d_R_xy_crust_mantle,
                 mp->d_R_xz_crust_mantle,
                 mp->d_R_yz_crust_mantle,
                 mp->d_c11store_crust_mantle,
                 mp->d_c12store_crust_mantle,
                 mp->d_c13store_crust_mantle,
                 mp->d_c14store_crust_mantle,
                 mp->d_c15store_crust_mantle,
                 mp->d_c16store_crust_mantle,
                 mp->d_c22store_crust_mantle,
                 mp->d_c23store_crust_mantle,
                 mp->d_c24store_crust_mantle,
                 mp->d_c25store_crust_mantle,
                 mp->d_c26store_crust_mantle,
                 mp->d_c33store_crust_mantle,
                 mp->d_c34store_crust_mantle,
                 mp->d_c35store_crust_mantle,
                 mp->d_c36store_crust_mantle,
                 mp->d_c44store_crust_mantle,
                 mp->d_c45store_crust_mantle,
                 mp->d_c46store_crust_mantle,
                 mp->d_c55store_crust_mantle,
                 mp->d_c56store_crust_mantle,
                 mp->d_c66store_crust_mantle,
                 mp->d_b_epsilondev_xx_crust_mantle,
                 mp->d_b_epsilondev_yy_crust_mantle,
                 mp->d_b_epsilondev_xy_crust_mantle,
                 mp->d_b_epsilondev_xz_crust_mantle,
                 mp->d_b_epsilondev_yz_crust_mantle,
                 mp->d_b_eps_trace_over_3_crust_mantle,
                 mp->d_b_R_xx_crust_mantle,
                 mp->d_b_R_yy_crust_mantle,
                 mp->d_b_R_xy_crust_mantle,
                 mp->d_b_R_xz_crust_mantle,
                 mp->d_b_R_yz_crust_mantle,
                 FORWARD_OR_ADJOINT);
  }

#ifdef ENABLE_VERY_SLOW_ERROR_CHECKING
  exit_on_gpu_error ("compute_forces_crust_mantle_ocl");
#endif
}
