!=====================================================================
!
!          S p e c f e m 3 D  G l o b e  V e r s i o n  4 . 0
!          --------------------------------------------------
!
!          Main authors: Dimitri Komatitsch and Jeroen Tromp
!    Seismological Laboratory, California Institute of Technology, USA
!             and University of Pau / CNRS / INRIA, France
! (c) California Institute of Technology and University of Pau / CNRS / INRIA
!                            February 2008
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

! compute several rheological and geometrical properties for a given spectral element
  subroutine compute_element_properties(ispec,iregion_code,idoubling, &
           xstore,ystore,zstore,nspec, &
           nspl,rspl,espl,espl2,ELLIPTICITY,TOPOGRAPHY,TRANSVERSE_ISOTROPY, &
           ANISOTROPIC_3D_MANTLE,ANISOTROPIC_INNER_CORE,ISOTROPIC_3D_MANTLE,CRUSTAL,ONE_CRUST, &
           myrank,ibathy_topo,ATTENUATION,ATTENUATION_3D, &
           ABSORBING_CONDITIONS, &
           RICB,RCMB,R670,RMOHO,RTOPDDOUBLEPRIME,R600,R220,R771,R400,R120,R80,RMIDDLE_CRUST,ROCEAN, &
           xelm,yelm,zelm,shape3D,dershape3D,rmin,rmax,rhostore,kappavstore,kappahstore,muvstore,muhstore,eta_anisostore, &
           xixstore,xiystore,xizstore,etaxstore,etaystore,etazstore,gammaxstore,gammaystore,gammazstore, &
           c11store,c12store,c13store,c14store,c15store,c16store,c22store, &
           c23store,c24store,c25store,c26store,c33store,c34store,c35store, &
           c36store,c44store,c45store,c46store,c55store,c56store,c66store, &
           nspec_ani,nspec_stacey,Qmu_store,tau_e_store,tau_s,T_c_source,rho_vp,rho_vs,&
           AM_V,AM_S,AS_V, &
           ACTUALLY_STORE_ARRAYS)

  implicit none

  include "constants.h"

! attenuation_model_variables
  type attenuation_model_variables
    sequence
    double precision min_period, max_period
    double precision                          :: QT_c_source        ! Source Frequency
    double precision, dimension(:), pointer   :: Qtau_s             ! tau_sigma
    double precision, dimension(:), pointer   :: QrDisc             ! Discontinutitues Defined
    double precision, dimension(:), pointer   :: Qr                 ! Radius
    integer, dimension(:), pointer            :: interval_Q                 ! Steps
    double precision, dimension(:), pointer   :: Qmu                ! Shear Attenuation
    double precision, dimension(:,:), pointer :: Qtau_e             ! tau_epsilon
    double precision, dimension(:), pointer   :: Qomsb, Qomsb2      ! one_minus_sum_beta
    double precision, dimension(:,:), pointer :: Qfc, Qfc2          ! factor_common
    double precision, dimension(:), pointer   :: Qsf, Qsf2          ! scale_factor
    integer, dimension(:), pointer            :: Qrmin              ! Max and Mins of idoubling
    integer, dimension(:), pointer            :: Qrmax              ! Max and Mins of idoubling
    integer                                   :: Qn                 ! Number of points
  end type attenuation_model_variables

  type (attenuation_model_variables) AM_V
! attenuation_model_variables

! attenuation_model_storage
  type attenuation_model_storage
    sequence
    integer Q_resolution
    integer Q_max
    double precision, dimension(:,:), pointer :: tau_e_storage
    double precision, dimension(:), pointer :: Qmu_storage
  end type attenuation_model_storage

  type (attenuation_model_storage) AM_S
! attenuation_model_storage

! attenuation_simplex_variables
  type attenuation_simplex_variables
    sequence
    integer nf          ! nf    = Number of Frequencies
    integer nsls        ! nsls  = Number of Standard Linear Solids
    double precision Q  ! Q     = Desired Value of Attenuation or Q
    double precision iQ ! iQ    = 1/Q
    double precision, dimension(:), pointer ::  f
    ! f = Frequencies at which to evaluate the solution
    double precision, dimension(:), pointer :: tau_s
    ! tau_s = Tau_sigma defined by the frequency range and
    !             number of standard linear solids
  end type attenuation_simplex_variables

  type(attenuation_simplex_variables) AS_V
! attenuation_simplex_variables

! correct number of spectral elements in each block depending on chunk type
  integer ispec,nspec,nspec_stacey

  logical ELLIPTICITY,TOPOGRAPHY
  logical TRANSVERSE_ISOTROPY,ANISOTROPIC_3D_MANTLE,ANISOTROPIC_INNER_CORE,ISOTROPIC_3D_MANTLE,CRUSTAL,ONE_CRUST

  logical ATTENUATION,ATTENUATION_3D,ABSORBING_CONDITIONS,ACTUALLY_STORE_ARRAYS

  double precision RICB,RCMB,R670,RMOHO, &
          RTOPDDOUBLEPRIME,R600,R220,R771,R400,R120,R80,RMIDDLE_CRUST,ROCEAN

! use integer array to store values
  integer ibathy_topo(NX_BATHY,NY_BATHY)

! arrays with the mesh in double precision
  double precision xstore(NGLLX,NGLLY,NGLLZ,nspec)
  double precision ystore(NGLLX,NGLLY,NGLLZ,nspec)
  double precision zstore(NGLLX,NGLLY,NGLLZ,nspec)

! code for the four regions of the mesh
  integer iregion_code

! 3D shape functions and their derivatives
  double precision, dimension(NGNOD,NGLLX,NGLLY,NGLLZ) :: shape3D
  double precision, dimension(NDIM,NGNOD,NGLLX,NGLLY,NGLLZ) :: dershape3D

! for ellipticity
  integer nspl
  double precision rspl(NR),espl(NR),espl2(NR)

  double precision, dimension(NGNOD) :: xelm,yelm,zelm

! parameters needed to store the radii of the grid points
! in the spherically symmetric Earth
  integer idoubling(nspec)
  double precision rmin,rmax

! for model density and anisotropy
  integer nspec_ani
  real(kind=CUSTOM_REAL), dimension(NGLLX,NGLLY,NGLLZ,nspec) :: rhostore,kappavstore,kappahstore,muvstore,muhstore,eta_anisostore

! the 21 coefficients for an anisotropic medium in reduced notation
  real(kind=CUSTOM_REAL), dimension(NGLLX,NGLLY,NGLLZ,nspec_ani) :: &
    c11store,c12store,c13store,c14store,c15store,c16store,c22store, &
    c23store,c24store,c25store,c26store,c33store,c34store,c35store, &
    c36store,c44store,c45store,c46store,c55store,c56store,c66store

! arrays with mesh parameters
  real(kind=CUSTOM_REAL), dimension(NGLLX,NGLLY,NGLLZ,nspec) :: xixstore,xiystore,xizstore, &
    etaxstore,etaystore,etazstore,gammaxstore,gammaystore,gammazstore

! proc numbers for MPI
  integer myrank

! Stacey, indices for Clayton-Engquist absorbing conditions
  real(kind=CUSTOM_REAL), dimension(NGLLX,NGLLY,NGLLZ,nspec_stacey) :: rho_vp,rho_vs

! attenuation
  double precision, dimension(NGLLX,NGLLY,NGLLZ,nspec) :: Qmu_store
  double precision, dimension(N_SLS,NGLLX,NGLLY,NGLLZ,nspec) :: tau_e_store
  double precision, dimension(N_SLS)                  :: tau_s
  double precision  T_c_source

! **************
! add topography on the Moho *before* adding the 3D crustal model so that the streched
! mesh gets assigned the right model values
  if(idoubling(ispec)==IFLAG_CRUST .or. idoubling(ispec)==IFLAG_220_80 &
     .or. idoubling(ispec)==IFLAG_80_MOHO) call add_moho_topography(myrank,xelm,yelm,zelm,RMOHO,R220)

! compute values for the Earth model
  call get_model(myrank,iregion_code,nspec, &
          kappavstore,kappahstore,muvstore,muhstore,eta_anisostore,rhostore,nspec_ani, &
          c11store,c12store,c13store,c14store,c15store,c16store,c22store, &
          c23store,c24store,c25store,c26store,c33store,c34store,c35store, &
          c36store,c44store,c45store,c46store,c55store,c56store,c66store, &
          xelm,yelm,zelm,shape3D,ispec, &
          rmin,rmax,idoubling(ispec),rho_vp,rho_vs,nspec_stacey, &
          TRANSVERSE_ISOTROPY,ANISOTROPIC_3D_MANTLE,ANISOTROPIC_INNER_CORE, &
          ISOTROPIC_3D_MANTLE,CRUSTAL,ONE_CRUST, &
          ATTENUATION, ATTENUATION_3D, tau_s, tau_e_store, Qmu_store, T_c_source, &
          size(tau_e_store,2), size(tau_e_store,3), size(tau_e_store,4), size(tau_e_store,5), &
          ABSORBING_CONDITIONS, &
          RCMB,RICB,R670,RMOHO,RTOPDDOUBLEPRIME,R600,R220,R771,R400,R120,R80,RMIDDLE_CRUST,ROCEAN,&
          AM_V,AM_S,AS_V)

! add topography without the crustal model
  if(TOPOGRAPHY .and. (idoubling(ispec)==IFLAG_CRUST .or. idoubling(ispec)==IFLAG_220_80 &
     .or. idoubling(ispec)==IFLAG_80_MOHO)) call add_topography(myrank,xelm,yelm,zelm,ibathy_topo,R220)

! add topography on 410 km and 650 km discontinuity in model S362ANI
  call add_topography_410_650(myrank,xelm,yelm,zelm,R220,R400,R670,R771)

! CMB topography
!  if(THREE_D_MODEL == THREE_D_MODEL_S362ANI .and. (idoubling(ispec)==IFLAG_MANTLE_NORMAL &
!     .or. idoubling(ispec)==IFLAG_OUTER_CORE_NORMAL)) &
!           call add_topography_cmb(myrank,xelm,yelm,zelm,RTOPDDOUBLEPRIME,RCMB)

! ICB topography
!  if(THREE_D_MODEL == THREE_D_MODEL_S362ANI .and. (idoubling(ispec)==IFLAG_OUTER_CORE_NORMAL &
!     .or. idoubling(ispec)==IFLAG_INNER_CORE_NORMAL .or. idoubling(ispec)==IFLAG_MIDDLE_CENTRAL_CUBE &
!     .or. idoubling(ispec)==IFLAG_BOTTOM_CENTRAL_CUBE .or. idoubling(ispec)==IFLAG_TOP_CENTRAL_CUBE &
!     .or. idoubling(ispec)==IFLAG_IN_FICTITIOUS_CUBE)) &
!           call add_topography_icb(myrank,xelm,yelm,zelm,RICB,RCMB)

! make the Earth elliptical
  if(ELLIPTICITY) call get_ellipticity(xelm,yelm,zelm,nspl,rspl,espl,espl2)

! recompute coordinates and jacobian for real 3-D model
  call calc_jacobian(myrank,xixstore,xiystore,xizstore, &
          etaxstore,etaystore,etazstore,gammaxstore,gammaystore,gammazstore, &
          xstore,ystore,zstore,xelm,yelm,zelm,shape3D,dershape3D,ispec,nspec,ACTUALLY_STORE_ARRAYS)

  end subroutine compute_element_properties
