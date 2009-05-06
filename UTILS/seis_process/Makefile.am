## Process this file with automake to produce Makefile.in

bin_PROGRAMS = asc2sac convolve_stf get_sac_header
bin_SCRIPTS = process_data.pl process_syn.pl rotate.pl

AM_CPPFLAGS = -I$(SACAUX)/../include
AM_LDFLAGS = -L$(SACAUX)/../lib

asc2sac_SOURCES = asc2sac.c
convolve_stf_SOURCES = convolve_stf.c
get_sac_header_SOURCES = get_sac_header.c
