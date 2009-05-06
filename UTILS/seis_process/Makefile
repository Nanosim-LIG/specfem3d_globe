# %License%



## The environment variable SACAUX is required for running SAC.
## We will assume SACAUX is already defined.

## The SAC library on GPS machine is for SUN, not for linux
## using my version for now
SACAUX = /home/tan2/opt/sac-101.2/aux

CC = gcc
CFLAGS = -O2
CPPFLAGS = -I$(SACAUX)/../include
LDFLAGS = -L$(SACAUX)/../lib -lsacio

all: asc2sac convolve_stf

asc2sac: asc2sac.c
	$(CC) asc2sac.c -o asc2sac $(CFLAGS) $(CPPFLAGS) $(LDFLAGS)

convolve_stf: convolve_stf.c
	$(CC) convolve_stf.c -o convolve_stf $(CFLAGS) $(CPPFLAGS) $(LDFLAGS) -lm


