/* License
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "sac.h"

int
main(int argc, char *argv[])
{
    int i;
    long int max, nlen, nerr;
    float yarray[1], beg, del;
    char *sacfn;

    if(argc < 3) {
	fprintf(stderr, "usage: %s sac-file var1 [var2 ...]\n", argv[0]);
	exit(1);
    }

    /* Read evenly spaced SAC file */
    /* The library does not allow reading header only, so we need to
     * read in some data as well... */
    sacfn = argv[1];
    max = 1;
    nerr = 0;
    rsac1(sacfn, yarray, &nlen, &beg, &del, &max, &nerr, strlen(sacfn));
    if(nerr && (nerr != -803)) {
        /* -803 means the file is longer than requested size. Ignored. */
	fprintf(stderr, "error when reading '%s', err=%d\n", sacfn, nerr);
	exit(-1);
    }

    /* Get header values */
    for(i=2; i<argc; i++) {
        long int itmp;
        float ftmp;
        char ctmp[33];
        char c;

        /* the 1st character must be ASCII */
        c = argv[i][0];
        if(c < 'A' || c > 'z') {
            fprintf(stderr, "argument '%s' is not a valid SAC header\n", argv[i]);
            exit(1);
        }
        fprintf(stderr, "argument is '%s'\n", argv[i]);

        /* the first character of SAC header indicated its type */
        switch(tolower(c)) {
        case 'n':
            /* integer type */
            getnhv(argv[i], &itmp, &nerr, strlen(argv[i]));
            printf("%10d", itmp);
            break;
        case 'i':
            /* enumerated type */
            getihv(argv[i], ctmp, &nerr, strlen(argv[i]), strlen(ctmp));
            printf("%s", ctmp);
            break;
        case 'l':
            /* logical type */
            getlhv(argv[i], &itmp, &nerr, strlen(argv[i]));
            printf("%d", itmp);
            break;
        case 'k':
            /* char string type */
            getkhv(argv[i], ctmp, &nerr, strlen(argv[i]), strlen(ctmp));
            printf("%s", ctmp);
            break;
        default:
            /* float type */
            getfhv(argv[i], &ftmp, &nerr, strlen(argv[i]));
            printf("%12.6g", ftmp);
            break;
        }

        if(nerr) {
            fprintf(stderr, "error when reading header of '%s'\n", sacfn);
            exit(-1);
        }

        /* each field is separated by at least two spaces */
        printf("  ");
    }

    printf("\n");

    return 0;
}
