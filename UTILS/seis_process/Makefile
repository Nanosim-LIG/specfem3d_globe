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

all: asc2sac get_sac_header

asc2sac: asc2sac.c
	$(CC) asc2sac.c -o asc2sac $(CFLAGS) $(CPPFLAGS) $(LDFLAGS)

get_sac_header: get_sac_header.c
	$(CC) get_sac_header.c -o get_sac_header $(CFLAGS) $(CPPFLAGS) $(LDFLAGS)


