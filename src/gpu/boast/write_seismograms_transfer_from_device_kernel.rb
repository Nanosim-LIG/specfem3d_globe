module BOAST

  def BOAST::write_seismograms_transfer_from_device_kernel(ref = true, n_gll3 = 125)
    BOAST::write_seismograms_from_device_kernel(:transfer, ref, n_gll3)
  end

  def BOAST::write_seismograms_from_device_kernel(type, ref = true, n_gll3 = 125)
    push_env( :array_start => 0 )
    kernel = CKernel::new

    if type == :transfer then
      function_name = "write_seismograms_transfer_from_device_kernel"
    elsif type == :transfer_scalar then
      function_name = "write_seismograms_transfer_scalar_from_device_kernel"
    else
      raise "Unsupported type : #{type}!"
    end

    number_receiver_global = Int( "number_receiver_global", :dir => :in, :dim => [Dim()] )
    ispec_selected_rec =     Int( "ispec_selected_rec",     :dir => :in, :dim => [Dim()] )
    ibool =                  Int( "ibool",                  :dir => :in, :dim => [Dim()] )
    station_seismo_field =   Real("station_seismo_field",   :dir => :out,:dim => [Dim()] )
    desired_field =          Real("desired_field",          :dir => :in,:dim => [Dim()] )
    nrec_local =             Int( "nrec_local",             :dir => :in)

    ngll3 =                  Int("NGLL3",                   :const => n_gll3)

    p = Procedure(function_name, [number_receiver_global,ispec_selected_rec,ibool,station_seismo_field,desired_field,nrec_local])
    if(get_lang == CUDA and ref) then
      @@output.print File::read("references/#{function_name}.cu")
    elsif(get_lang == CL or get_lang == CUDA) then
      make_specfem3d_header( :ngll3 => n_gll3 )
      decl p
      blockID    = Int("blockID")
      iglob      = Int("iglob")
      irec       = Int("irec")
      ispec      = Int("ispec")
      decl blockID
      print blockID === get_group_id(0)+get_group_id(1)*get_num_groups(0)
      print If(blockID<nrec_local) {
        decl irec, ispec, iglob
        print irec === number_receiver_global[blockID] - 1
        print ispec === ispec_selected_rec[irec] - 1
        print iglob === ibool[get_local_id(0) + ngll3*ispec] - 1
        if type == :transfer then
          (0..2).each { |i|
            print station_seismo_field[ngll3*3*blockID + get_local_id(0)*3 + i] === desired_field[iglob*3+i]
          }
        else
          print station_seismo_field[ngll3*blockID + get_local_id(0)] === desired_field[iglob];
        end
      }

      close p
    else
      raise "Unsupported language!"
    end
    pop_env( :array_start )
    kernel.procedure = p
    return kernel
  end
end

