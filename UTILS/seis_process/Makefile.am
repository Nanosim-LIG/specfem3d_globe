## Process this file with automake to produce Makefile.in

SUFFIXES = .pl.in .pl

bin_PROGRAMS = asc2sac convolve_stf
bin_SCRIPTS = process_data.pl process_syn.pl rotate.pl

AM_CPPFLAGS = -I$(SACAUX)/../include
AM_LDFLAGS = -L$(SACAUX)/../lib

asc2sac_SOURCES = asc2sac.c
convolve_stf_SOURCES = convolve_stf.c


do_build = sed -e s:[@]sacaux[@]:$(SACAUX):g -e s:[@]prefix[@]:$(prefix):g

.pl.in.pl:
	$(do_build) $< > $@ || (rm -f $@ && exit 1)

CLEANFILES = $(bin_SCRIPTS)
EXTRA_DIST = process_data.pl.in process_syn.pl.in rotate.pl.in
