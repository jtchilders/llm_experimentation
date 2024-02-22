#!/bin/bash
#PBS -l walltime=01:00:00
#PBS -l select=4
#PBS -A datascience
#PBS -l filesystems=home:eagle
#PBS -q debug-scaling
#PBS -o results/submit_polaris.stdout
#PBS -e results/submit_polaris.stderr

cd $PBS_O_WORKDIR
export PBS_JOBID_NUMBER=$(echo $PBS_JOBID | cut -d'.' -f1)
export FILENAME_DATE=$(date +"%Y-%m-%d_%H%M").jid${PBS_JOBID_NUMBER}
echo starting job: $FILENAME_DATE
{
   
   module load conda/2023-10-04
   conda activate

   echo listing nodes:
   cat $PBS_NODEFILE

   NUM_NODES=$(cat $PBS_NODEFILE| wc -l)
   NUM_GPUS_PER_NODE=4
   TOTAL_MPI_RANKS=$((NUM_GPUS_PER_NODE * NUM_NODES))
   echo TOTAL_MPI_RANKS: $TOTAL_MPI_RANKS
   echo NUM_GPUS_PER_NODE: $NUM_GPUS_PER_NODE
   echo NUM_NODES: $NUM_NODES
   export HDF5_DISABLE_VERSION_CHECK=1
   mpiexec -n $TOTAL_MPI_RANKS -ppn $NUM_GPUS_PER_NODE --hostfile $PBS_NODEFILE python main.py -o results/$FILENAME_DATE -b 4096 -c 10 -e 1000 -i 50 -l 0.05 --warmup-steps 1000 --total-steps 3000 --train-file /lus/eagle/projects/datascience/parton/data/wikitext-103-raw/wiki.train.raw.tokens.symmetric.c2.h5

} 1> results/$FILENAME_DATE.out 2> results/$FILENAME_DATE.err
echo finished job: $FILENAME_DATE  time: $SECONDS