/*
 * asc2sac.c by Eh Tan
 *
 * Copyright (C) 2009, California Institute of Technology.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <limits.h>
#include <errno.h>
#include "sac.h"

int
main(int argc, char *argv[])
{
    long int npts, nerr, i;
    long int itmp;
    float ftmp;
    char *endptr, *str;
    char *ascfn, *sacfn;
    float *time, *data;
    FILE *f;

    if(argc < 4) {
        fprintf(stderr,
                "Converting evenly spaced time series ASCII file to SAC binary\n"
                "usage: %s ascii-file npts sac-file\n\n"
                "ascii-file: name of the input file. The file must have two columns,\n"
                "            1st column must be evenly spaced.\n"
                "npts:       number of lines in the input file.\n"
                "sac-file:   name of the output file.\n",
                argv[0]);
        exit(1);
    }

    str = argv[2];
    errno = 0;
    endptr = str;
    npts = strtol(str, &endptr, 10);

    /* Check for various possible errors */
    if (errno || (*endptr != 0)) {
        fprintf(stderr, "Error when converting '%s' to integer\n", str);
        exit(-1);
    }

    if (npts <= 0) {
        fprintf(stderr, "npts must be positive\n");
        exit(EXIT_FAILURE);
    }

    /* If we got here, strtol() successfully parsed a number */
    ascfn = argv[1];
    sacfn = argv[3];

    time = (float*) malloc(npts * sizeof(float));
    data = (float*) malloc(npts * sizeof(float));

    if(time == NULL || data == NULL) {
        fprintf(stderr, "Out of memory\n");
        exit(1);
    }


    /* reading ascii file */
    f = fopen(ascfn, "r");
    if(f == NULL) {
	fprintf(stderr, "Cannot open file '%s' to read\n", ascfn);
	exit(-1);
    }

    for(i=0; i<npts; i++) {
        char buffer[255];
        float a, b;
        fgets(buffer, 254, f);
        if(sscanf(buffer, "%f %f\n", &a, &b) != 2) {
            fprintf(stderr, "error when reading file '%s'\n", ascfn);
            exit(-1);
        }
        time[i] = a;
        data[i] = b;
    }
    fclose(f);
    /* finished reading ascii file */

    /* write SAC data usng SAC IO library */
    nerr = 0;
    newhdr();
    setnhv("npts", &npts, &nerr, strlen("npts"));
    itmp = 1;
    setlhv("leven", &itmp, &nerr, strlen("leven"));
    ftmp = time[1] - time[0];
    setfhv("delta", &ftmp, &nerr, strlen("delta"));
    setfhv("b", &(time[0]), &nerr, strlen("b"));
    setfhv("e", &(time[npts-1]), &nerr, strlen("e"));
    setihv("iftype", "itime", &nerr, strlen("iftype"), strlen("itime"));
    setihv("idep", "idisp", &nerr, strlen("idep"), strlen("idisp"));

    if(nerr) {
	fprintf(stderr, "error when setting header for '%s'\n", sacfn);
        exit(-1);
    }

    wsac0(sacfn, time, data, &nerr, strlen(sacfn));

    if(nerr) {
        fprintf(stderr, "error when writing '%s'\n", sacfn);
        exit(-1);
    }

    return 0;
}
