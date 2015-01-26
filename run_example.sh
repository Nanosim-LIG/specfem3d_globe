#! /bin/bash

SPECFEM_BUILD=/home/kpouget/specfem3d_build
SPECFEM_DIR=/home/kpouget/specfem3d_globe
SPECFEM_DIR_ABS=$(readlink -f $SPECFEM_DIR)

EXAMPLE_DIR=$SPECFEM_DIR_ABS/EXAMPLES/regional_Greece_small
#EXAMPLE_DIR=$SPECFEM_DIR_ABS/EXAMPLES/global_s362ani_small

cd $SPECFEM_BUILD
WORKDIR=$(pwd)

echo "Example directory: $EXAMPLE_DIR"
echo

get_config() {
    CONF="DATA/Par_file"
    BASE_MPI_DIR=`grep LOCAL_PATH $CONF | cut -d= -f2`
    NPROC_XI=`grep NPROC_XI $CONF       | cut -d= -f2`
    NPROC_ETA=`grep NPROC_ETA $CONF     | cut -d= -f2`
    NCHUNKS=`grep NCHUNKS $CONF         | cut -d= -f2`
    
    NUM_NODES=$(( $NCHUNKS * $NPROC_XI * $NPROC_ETA ))
}

prepare() {
    echo
    echo "   setting up example..."
    echo
    
    mkdir -p DATABASES_MPI
    mkdir -p OUTPUT_FILES
    mkdir -p DATA

    ln -sf $SPECFEM_DIR_ABS/DATA/crust2.0 DATA
    ln -sf $SPECFEM_DIR_ABS/DATA/s362ani DATA
    ln -sf $SPECFEM_DIR_ABS/DATA/QRFSI12 DATA
    ln -sf $SPECFEM_DIR_ABS/DATA/topo_bathy DATA

    cp $EXAMPLE_DIR/DATA/{Par_file,STATIONS,CMTSOLUTION} DATA -v

    echo "Preparation done, you can build and run now the example"
}

mesh() {
    get_config
    if [[ ! -f .mesh_created || DATA/Par_file -nt .mesh_created ]]
    then 
        echo "Preparing mesher script on $NUM_NODES processors." >&2

        JOB_NAME=xmeshfem3d
    
        cat > job_meshfem.sh <<EOF
#!/bin/sh

#SBATCH --partition=mb
#SBATCH --ntasks=$NUM_NODES
#SBATCH --cpus-per-task=2
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=$JOB_NAME
#SBATCH --error=batch/$JOB_NAME.err
#SBATCH --output=batch/$JOB_NAME.out
#SBATCH --workdir=$WORKDIR

mpirun bin/xmeshfem3D
date > .mesh_created

EOF
        rm batch/$JOB_NAME.{err,out} -f
        echo "job_meshfem.sh"
        
    else
        echo "mesh already generated, skip mesher" >&2
        echo "(remove .mesh_created to force regeneration)" >&2
    fi
}

specfem() {
    get_config
    
    JOB_NAME=specfem3d
    
    cat > job_specfem.sh <<EOF
#!/bin/sh

#SBATCH --partition=mb
#SBATCH --ntasks=$NUM_NODES
#SBATCH --cpus-per-task=2
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=$JOB_NAME
#SBATCH --error=batch/$JOB_NAME.err
#SBATCH --output=batch/$JOB_NAME.out
#SBATCH --workdir=$WORKDIR
#SBATCH --gres=gpu

mpirun bin/xspecfem3D

EOF
    rm batch/$JOB_NAME.{err,out} -f
    
    echo "job_specfem.sh"
}

enqueue() {
    JOB=$1
    DEPENDS=$2
    
    if [[ ! -f $JOB ]]; then
        echo "ERROR: Cannot enqueue job '$JOB', it's not a file." >&2
        exit
    fi

    if [[ ! -z  $DEPENDS ]]; then
        OPT="--dependency=afterok:$DEPENDS" 
        echo "Job $JOB depends of $DEPENDS." >&2
    fi

    res=$(sbatch $OPT $JOB)
    job_id=$(echo $res | sed 's/Submitted batch job //')
    echo "Job $JOB enqueued with id=$job_id." >&2
    echo $job_id
}

while test ${#} -gt 0
do
  if [ $1 == 'prepare' ]
  then
      prepare
  elif [ $1 == 'mesh' ]
  then
      mesh
  elif [ $1 == 'specfem' ]
  then
      specfem
  elif [ $1 == 'go' ]
  then
      echo "Building specfem ..."
      make || exit 1
      mesh_job=$(mesh)
      if [[ ! -z $mesh_job ]]; then
          mesh_id=$(enqueue $mesh_job)
      fi
      spec_id=$(enqueue $(specfem) $mesh_id)
      echo "Specfem job enqueued with id=$spec_id."
  else
      echo "Argument $1 not recognized"
  fi
  shift
done
