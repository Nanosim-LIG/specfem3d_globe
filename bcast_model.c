
#include <mpi.h>

#include <stdio.h>
#include <stdlib.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <unistd.h>

#include "config.h"


static char wd[150], modelDir[150];


void FC_FUNC_(bcast_model, BCAST_MODEL)(int *pRank, char *scratchDir, int scratchDirLen) {
    int fd, rank;
    struct stat statBuf;
    int size;
    char *data;
    int status;
    
    rank = *pRank;
    
    /* Save the working directory (which is on the shared filesystem)
       for future reference. */
    if (!getcwd(wd, sizeof(wd))) {
        perror("getcwd");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    /* Broadcast the model archive to all the nodes. */
    if (rank == 0) {
        /* XXX: We shouldn't hardcode this filename. */
        fd = open("bcast_model.tgz", O_RDONLY);
        if (fd == -1) {
            perror("open");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (fstat(fd, &statBuf) == -1) {
            perror("fstat");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        size = (int)statBuf.st_size;
    }
    status = MPI_Bcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (status != MPI_SUCCESS) {
        fprintf(stderr, "MPI_Bcast: error %d\n", status);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    data = (char *)malloc(size);
    if (!data) {
        perror("malloc");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (rank == 0) {
        if (read(fd, data, size) == -1) {
            perror("read");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        close(fd);
    }
    MPI_Bcast(data, size, MPI_CHAR, 0, MPI_COMM_WORLD);
    if (status != MPI_SUCCESS) {
        fprintf(stderr, "MPI_Bcast: error %d\n", status);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    /* Create and enter the model directory. */
    sprintf(modelDir, "%.*s/model-%d", scratchDirLen, scratchDir, rank);
    if (mkdir(modelDir, 0777) == -1) {
        perror("mkdir");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (chdir(modelDir) == -1) {
        perror("chdir");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    /* Save a local copy of the model archive. */
    fd = open("bcast_model.tgz", O_CREAT | O_WRONLY, S_IRUSR);
    if (fd == -1) {
        perror("open");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (write(fd, data, size) == -1) {
        perror("write");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    close(fd);
    free(data);
    
    /* Extract the model files. */
    status = system("tar xzf bcast_model.tgz");
    if (status == -1) {
        perror("system");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    status = WEXITSTATUS(status);
    if (status != 0) {
        fprintf(stderr, "tar: exit %d\n", status);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    /* Stay in the model directory, ready to read files. */
}

void FC_FUNC_(enter_model_dir, ENTER_MODEL_DIR)() {
    if (chdir(modelDir) == -1) {
        perror("chdir");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

void FC_FUNC_(leave_model_dir, LEAVE_MODEL_DIR)() {
    if (chdir(wd) == -1) {
        perror("chdir");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
