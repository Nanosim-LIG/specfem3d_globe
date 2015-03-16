# /bin/bash

SPECFEM_BUILD=$HOME/specfem3d_build
SPECFEM_DIR=$HOME/specfem3d_globe
SPECFEM_DIR_ABS=$(readlink -f $SPECFEM_DIR)

EXAMPLE_DIR=$SPECFEM_DIR_ABS/EXAMPLES/regional_Greece_small/DATA
#EXAMPLE_DIR=$SPECFEM_DIR_ABS/EXAMPLES/small_benchmark_run_to_test_very_simple_Earth/
#EXAMPLE_DIR=$SPECFEM_DIR_ABS/EXAMPLES/global_s362ani_small/DATA
#EXAMPLE_DIR=$SPECFEM_DIR_ABS/EXAMPLES/regional_MiddleEast/DATA

cd $SPECFEM_BUILD
WORKDIR=$(pwd)

#echo "Example directory: $EXAMPLE_DIR"

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

    cp $EXAMPLE_DIR/{Par_file,STATIONS,CMTSOLUTION} DATA

	echo >> DATA/Par_file
	echo "# example dir is $EXAMPLE_DIR" >> DATA/Par_file

	mv DATA/Par_file DATA/Par_file.model

	adjust

    echo "Preparation done, you can build and run now the example"
}

adjust() {
    if [ -e DATA/Par_file.model ]
    then
        cat DATA/Par_file.model > DATA/Par_file
    fi
    source ~/Par_file.config
    for l in $(cat ~/Par_file.config)
    do
        key=$(echo $l | cut -d= -f1)
        val=$(echo $l | cut -d= -f2)
        sed "s|^\($key\s*=\).*$|\1 $val|" -i DATA/Par_file
    done
}

SPEC_RUN=$HOME/specfem3d_run
SPEC_BUILD=$HOME/specfem3d_build

specfem() {
    get_config
    echo "Preparing specfem script on $NUM_NODES processors." | grep --color=always -e '^Preparing.*$' >&2
    if [[ -z "$1" ]]
    then
        JOB_NAME=specfem
    else
        JOB_NAME=$1
    fi
    
    cat > job_$JOB_NAME.sh <<EOF
#!/bin/bash

#SBATCH --partition=mb
#SBATCH --ntasks=$NUM_NODES
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=$JOB_NAME
#SBATCH --error=$SPEC_RUN/%j/log_$JOB_NAME.err
#SBATCH --output=$SPEC_RUN/%j/log_$JOB_NAME.out
#SBATCH --workdir=$WORKDIR
#SBATCH --gres=gpu

###########################
### Prepare environment ###
###########################

function cleanup {
    date

    cp -r $SPEC_BUILD/OUTPUT_FILES $SPEC_RUN/\$SLURM_JOB_ID -r
    DEST="$SPEC_RUN/\$(date --iso-8601=minutes | cut -d+ -f1 | sed 's/://' | sed s/T/_/)_\$what-\$?"
    mv  $SPEC_RUN/\$SLURM_JOB_ID \$DEST
    rm current last -f
    ln -s \$DEST last
}

cd $SPEC_BUILD/OUTPUT_FILES/
rm -rf *.ascii timestamp_forward* *.vtk gpu_device* output_solver.txt starttimeloop.txt output_list_stations.txt
cd - > /dev/null

echo "Checking specfem build state ..."
do_make=\$(check_make)
if [ \$? != 0 ]
then
  echo "Specfem build is not up to date, please recompile ..."
  echo \$do_make 
  exit 1
fi

#############################
### Select execution type ###
#############################


run_mesh=\$([[ ! -f .mesh_created || DATA/Par_file -nt .mesh_created || ! -f DATABASES_MPI/addressing.txt ]])\$?

what=\$([[ \$run_mesh == 0 ]] && echo meshfem || echo specfem)

trap cleanup EXIT

if [[ ! "$JOB_NAME" -eq "\$what" ]]
then
  echo "Cannot run $JOB_NAME. Execution CANCELLED"
  [[ ! -f .mesh_created ]] && echo "File .mesh_created not found" 
  [[ DATA/Par_file -nt .mesh_created ]] && echo "DATA/Par_file newer than .mesh_created"
  [[ ! -f DATABASES_MPI/addressing.txt ]] && echo "File DATABASES_MPI/addressing.txt not found"

  exit 1
fi

#########################
### Trigger execution ###
#########################

if [[ \$run_mesh == 0 ]]
then
  echo '########################' | tee -a /dev/stderr
  echo '#### Running Meshfem ###' | tee -a /dev/stderr
  echo '########################' | tee -a /dev/stderr

  time mpirun bin/xmeshfem3D 
  if [[ \$? == 0 ]]
  then
     mv OUTPUT_FILES/addressing.txt DATABASES_MPI/

     date > .mesh_created
     echo --ntasks=$NUM_NODES >> .mesh_created
     echo Mesh done.
     how=0
  else
     how=\$?
     echo "ERROR: xmeshfem3D failed (errcode=\$how)"

     touch bin/xmeshfem3D
     rm .mesh_created -f
  fi
else
  echo '########################' | tee -a /dev/stderr
  echo '#### Running Specfem ###' | tee -a /dev/stderr
  echo '########################' | tee -a /dev/stderr

  cp DATABASES_MPI/addressing.txt OUTPUT_FILES/ -f

  time mpirun bin/xspecfem3D
  how=\$?
fi

exit \$how

EOF
    
    echo "job_$JOB_NAME.sh"
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
    chmod u+x $JOB
    my_sbatch $OPT $JOB
}
REMOTE="ssh.hca.bsc.es"
REMOTE="localhost"
compile() {
    if ssh $REMOTE "bash -c 'source ~/.profile ; \
                      cd ~/specfem3d_build && \
                      touch DATA/Par_file && \
                      make -j12'" &> $SPEC_RUN/specfem3d_compile.log
    then
        echo "Make successed"
        return 0
    else
        echo "Make failed ... :'( Check $SPEC_RUN/specfem3d_compile.log"
	return 1
    fi
}

run_benchmark() {    
    echo "Prepare mesher for $1"
    # run mesher
    if ! compile
    then
        echo "Couldn't compile, abort mesher run for $1"
        return 1
    fi
    echo "Run mesher for $1"
    run_example.sh go wait silent
	
    OUT="$(run_example.sh see |& grep 'CANCELLED')"
    if [[ ! -z "$OUT" ]]
    then
	echo "Execution for $1 cancelled ..."
	echo "$OUT"
	return 1
    fi
    # run mesher
    echo "Prepare specfem for $1"
    if ! compile
    then
        echo "Couldn't compile, abort specfem run for $1"
        return 1
    fi
    
    echo "Run specfem for $1"
    touch .mesh_created
    run_example.sh go wait silent
    
    echo "Done for $1"

    return 0
}

benchmark() {
    if [[ -e ~/specfem3d_run && ! -z "$(ls -A ~/specfem3d_run)" ]]
    then
        echo "Old run dir backup"
        mv ~/specfem3d_run{,.$(date --iso-8601=minutes)}
    fi
    
    mkdir ~/specfem3d_run -p
    cd $SPECFEM_BUILD
    
	exec > >(tee $SPEC_RUN/specfem3d_benchmark.log)
    exec 2>&1

    NEX=$(echo $(grep NEX DATA/Par_file | head -1 | cut -d= -f2))a

	# comment to compute dynamically
	NPROC_VALUES="4 8 16 32"
	if [[ -z $NPROC_VALUES ]]
	then
		echo "Computing NPROC values with DATA/Par_File::NEX=$NEX"
    	NPROC_VALUES=$(specfem_nproc_per_nex $NEX)
	fi
	echo "Benchmarking specfem for NPROC=$NPROC_VALUES"

    for val in $NPROC_VALUES
    do
        echo "Run benchmark with NPROC=$val"
        sed "s|^\(NPROC_XI\s*=\).*$|\1 $val|" -i DATA/Par_file
        sed "s|^\(NPROC_ETA\s*=\).*$|\1 $val|" -i DATA/Par_file
        
        run_benchmark $val
    done
}

txtblk='\e[0m' # Default
txtred='\e[0;31m' # Red

while test ${#} -gt 0
do
  if [ $1 == 'prepare' ]
  then
      prepare
  elif [ $1 == 'adjust' ]
  then
      adjust
  elif [ $1 == 'benchmark' ]
  then
      benchmark
  elif [ $1 == 'specfem' ]
  then
      specfem
  elif [ $1 == 'see' ]
  then
      cat last/specfem3d.out
      echo "============"
      cat last/specfem3d.err
      echo "============"
      file last
      
  elif [ $1 == 'wait' ]
  then
      if [ "$2" == 'silent' ]
      then
          SILENT=1
          echo "Let's wait in silence ..."
          shift
      fi
      cnt=0
	  tid=$(readlink current)
	  tid=${tid##*/}
	  echo "Waiting for task $tid ..."
      while squeue | grep -q $tid
      do
          last="$(squeue -h | grep kpouget | sed 's/\[.*\]/[...]/' | sed 's/ \+ / /g') ($cnt s)"
          [[ -z $SILENT ]] && echo -en "$last\r"
          sleep 1
          cnt=$((cnt + 1))
      done
      echo $last # reprint last message for logs, previous one got erased
      
  elif [ $1 == 'go' ]
  then
      run_mesh=$([[ ! -f .mesh_created || DATA/Par_file -nt .mesh_created || ! -f DATABASES_MPI/addressing.txt ]])$?
      what=$([[ $run_mesh == 0 ]] && echo meshfem || echo specfem)
      echo -e "### $txtred Running $what $txtblk ###" 
      
      echo "Checking specfem build state ..."
      do_make=$(check_make)
      if [ $? != 0 ]
      then
          echo "Specfem build is not  up to date, please recompile ..."
          echo $do_make 
          exit 1
      fi
      
      enqueue $(specfem $what)
  else
      echo "Argument $1 not recognized"
      exit 1
  fi
  shift
done
