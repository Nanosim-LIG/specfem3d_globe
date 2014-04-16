require 'rubygems'
require 'BOAST'
require 'narray'
require './HEADER.rb'

def rndup( val, div)
  return (val%div) == 0 ? val : val + div - (val%div)
end

$options = {:output_dir => "./output"}

$parser = OptionParser::new do |opts|
  opts.on("-c","--check","Check kernels by building them") {
    $options[:check] = true
  }
  opts.on("-d","--display","Display kernels") {
    $options[:display] = true
  }
  opts.on("-v","--verbose","Verbosity") {
    $options[:verbose] = true
  }
  opts.on("-o","--output-dir DIR","Output directory") { |dir|
    $options[:output_dir] = dir
  }
  opts.parse!
end

kernels = [
:assemble_boundary_accel_on_device,
:assemble_boundary_potential_on_device,
:prepare_boundary_potential_on_device,
:prepare_boundary_accel_on_device,
:get_maximum_scalar_kernel,
:get_maximum_vector_kernel,
:compute_add_sources_adjoint_kernel,
:compute_add_sources_kernel,
:compute_coupling_fluid_CMB_kernel,
:compute_coupling_fluid_ICB_kernel,
:compute_coupling_CMB_fluid_kernel,
:compute_coupling_ICB_fluid_kernel,
:compute_coupling_ocean_kernel,
:write_seismograms_transfer_from_device_kernel,
:write_seismograms_transfer_scalar_from_device_kernel,
:noise_transfer_surface_to_host_kernel,
:noise_add_source_master_rec_kernel,
:noise_add_surface_movie_kernel,
:compute_stacey_acoustic_kernel,
:compute_stacey_acoustic_backward_kernel,
:compute_stacey_elastic_kernel,
:compute_stacey_elastic_backward_kernel,
:update_disp_veloc_kernel,
:update_potential_kernel,
:update_accel_elastic_kernel,
:update_veloc_elastic_kernel,
:update_accel_acoustic_kernel,
:update_veloc_acoustic_kernel,
:outer_core_impl_kernel,
:inner_core_impl_kernel,
:compute_rho_kernel,
:compute_iso_kernel,
:compute_ani_kernel,
:compute_hess_kernel,
:compute_acoustic_kernel,
:compute_strength_noise_kernel,
:crust_mantle_impl_kernel
]

langs = [ :CUDA, :CL]
BOAST::set_default_real_size(4)
BOAST::set_replace_constants(false)
class Float
  alias_method :old_to_s, :to_s
  def to_s
    s = ""+old_to_s
    s += "f" if BOAST::get_default_real_size == 4
    return s
  end
end

kernels.each { |kern|
  require "./#{kern.to_s}.rb"
  puts kern.to_s
  langs.each { |lang|
    puts lang.to_s
    BOAST::set_lang( BOAST::const_get(lang))
    puts "REF" if lang == :CUDA
    k = BOAST::method(kern).call
    k.print if $options[:display]
    if lang == :CUDA then
      puts "Generated"
      k = BOAST::method(kern).call(false)
      k.print if $options[:display]
      filename = "#{kern}.cu"
    elsif lang == :CL
      filename = "#{kern}_cl.c"
    end
    f = File::new("#{$options[:output_dir]}/#{filename}", "w+")
    if lang == :CUDA then
      f.puts k
      k.build( :LDFLAGS => " -L/usr/local/cuda-5.5.22/lib64", :NVCCFLAGS => "-arch sm_20 -O2 --compiler-options -Wall", :verbose => $options[:verbose] ) if $options[:check]
    elsif lang == :CL then
      s = k.to_s
      res = "const char * #{kern}_program = \"\\\n"
      s.each_line { |line|
        res += line.sub("\n","\\n\\\n")
      }
      res += "\";\n"
      f.print res
      k.build(:verbose => $options[:verbose] ) if $options[:check]
    end
    f.close
  }
}

langs.each { |lang|
  if lang == :CUDA then
    suffix = ".cu"
    kern_proto_f = File::new("#{$options[:output_dir]}/kernel_proto.cu.h", "w+")
  elsif lang == :CL
    suffix = "_cl.c"
  end
  kern_inc_f = File::new("#{$options[:output_dir]}/kernel_inc"+suffix, "w+")
  kern_list_f = File::new("#{$options[:output_dir]}/kernel_list.h", "w+")

  kernels.each { |kern|
    kern_list_f.puts "BOAST_KERNEL(#{kern.to_s});"
    kern_inc_f.puts "#include \"#{kern.to_s}#{suffix}\""
    if lang == :CUDA then
      require "./#{kern.to_s}.rb"
      BOAST::set_lang( BOAST::const_get(lang))
      k = BOAST::method(kern).call(false)
      proto = k.procedure.decl(false)[0..-3]+";"
      kern_proto_f.puts proto
    end
  }
  kern_list_f.close
  kern_inc_f.close
  if lang == :CUDA then
    kern_proto_f.close
  end
}
