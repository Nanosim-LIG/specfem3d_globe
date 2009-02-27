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


module model_plugin_support_module

  include "constants.h"

  type model_plugin_support_variables
     integer NCHUNKS, NPROC_XI, NPROC_ETA
     double precision ANGULAR_WIDTH_XI_RAD, ANGULAR_WIDTH_ETA_RAD
     double precision, dimension(NDIM,NDIM) :: rotation_matrix
  end type model_plugin_support_variables

  type (model_plugin_support_variables) vars

end module model_plugin_support_module


subroutine get_nchunks(NCHUNKS)

  use model_plugin_support_module
  implicit none

  integer NCHUNKS

  NCHUNKS = vars%NCHUNKS

end subroutine get_nchunks


subroutine get_nproc(NPROC_XI, NPROC_ETA)

  use model_plugin_support_module
  implicit none

  integer NPROC_XI, NPROC_ETA

  NPROC_XI = vars%NPROC_XI
  NPROC_ETA = vars%NPROC_ETA

end subroutine get_nproc


subroutine get_angular_width_in_radians(ANGULAR_WIDTH_XI_RAD, ANGULAR_WIDTH_ETA_RAD)

  use model_plugin_support_module
  implicit none

  double precision ANGULAR_WIDTH_XI_RAD, ANGULAR_WIDTH_ETA_RAD

  ANGULAR_WIDTH_XI_RAD = vars%ANGULAR_WIDTH_XI_RAD
  ANGULAR_WIDTH_ETA_RAD = vars%ANGULAR_WIDTH_ETA_RAD

end subroutine get_angular_width_in_radians


subroutine get_rotation_matrix(rotation_matrix)

  use model_plugin_support_module
  implicit none

  double precision, dimension(NDIM,NDIM) :: rotation_matrix

  rotation_matrix = vars%rotation_matrix

end subroutine get_rotation_matrix


!---------------------------------------------------------------------
! private

subroutine init_model_plugin_support(NCHUNKS, NPROC_XI, NPROC_ETA, &
     ANGULAR_WIDTH_XI_RAD, ANGULAR_WIDTH_ETA_RAD, &
     rotation_matrix)

  use model_plugin_support_module
  implicit none

  integer NCHUNKS, NPROC_XI, NPROC_ETA
  double precision ANGULAR_WIDTH_XI_RAD, ANGULAR_WIDTH_ETA_RAD
  double precision, dimension(NDIM,NDIM) :: rotation_matrix

  vars%NCHUNKS = NCHUNKS

  vars%NPROC_XI = NPROC_XI
  vars%NPROC_ETA = NPROC_ETA

  vars%ANGULAR_WIDTH_XI_RAD = ANGULAR_WIDTH_XI_RAD
  vars%ANGULAR_WIDTH_ETA_RAD = ANGULAR_WIDTH_ETA_RAD

  vars%rotation_matrix = rotation_matrix

end subroutine init_model_plugin_support
