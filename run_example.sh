#! /bin/bash

SPECFEM_DIR=$1
SPECFEM_DIR_ABS=$(readlink -f $SPECFEM_DIR)

EXAMPLE_DIR=$SPECFEM_DIR_ABS/EXAMPLES/regional_Greece_small
#EXAMPLE_DIR=$SPECFEM_DIR_ABS/EXAMPLES/global_s362ani_small
date
echo "directory: $EXAMPLE_DIR"
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

    cp $EXAMPLE_DIR/DATA/{Par_file,STATIONS,CMTSOLUTION} DATA

    echo "Preparation done, you can build and run now the example"
}

build() {
    get_config
    if [ ! -e .mesh_created -o .mesh_created -ot DATA/Par_file ]
    then 
        echo "starting MPI mesher on $NUM_NODES processors (`date`)"
        mpirun -np $NUM_NODES bin/xmeshfem3D
        echo "mesher done: `date`"
        echo
        touch .mesh_created
    else
        echo "mesh already generated, skip mesher"
        echo "(remove .mesh_created to force regeneration)"
    fi
}

run() {
    get_config

    echo "starting run (`date`)"
    mpirun -np $NUM_NODES bin/xspecfem3D
    ret=$?
    if [ $ret == 0 ]
    then
        echo "finished successfully (`date`)"
    else
        echo "failed (`date`), errcode=$ret"
    fi
}
shift
while test ${#} -gt 0
do
  if [ $1 == 'run' ]
  then
      run
  elif [ $1 == 'build' ]
  then
      build
  elif [ $1 == 'prepare' ]
  then
      prepare
  else
      echo "Argument $1 not recognized"
  fi
  shift
done
