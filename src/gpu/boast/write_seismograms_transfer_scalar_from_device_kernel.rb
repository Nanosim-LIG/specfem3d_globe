require './write_seismograms_transfer_from_device_kernel.rb'

module BOAST

  def BOAST::write_seismograms_transfer_scalar_from_device_kernel(ref = true, n_gll3 = 125)
    BOAST::write_seismograms_from_device_kernel(:transfer_scalar, ref, n_gll3)
  end

end
