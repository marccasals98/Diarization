#!/bin/bash
#SBATCH -D .
#SBATCH -A bsc88
#SBATCH -q acc_debug
############# Obligatorias #######################
#SBATCH --time=0-02:00:00               # Consultar batchlim para entender los límites de las particiones. 
################# HOST ###########################
#SBATCH --nodes=1                       # Número de nodos
#SBATCH --ntasks=1                      # Número de tareas MPI totales
#SBATCH --ntasks-per-node=1             # Número de tareas MPI por nodo
#SBATCH --cpus-per-task=40              # Número de cores por tarea. Threads. $SLURM_CPUS_PER_TASK
#SBATCH --gres=gpu:2
################ Logging #########################
#SBATCH --job-name=diarization
#SBATCH --verbose
#SBATCH --output=/gpfs/projects/bsc88/speech/speaker_recognition/outputs/diarization/logs/%x_%j.txt


date +%Y-%m-%d_%H:%M:%S

# activate env
source /gpfs/projects/bsc88/speech/environments/diarize_venv/bin/activate

export TORCH_HOME="/gpfs/projects/bsc88/speech/speaker_recognition/outputs/diarization/cache/torch"

srun python src/train.py \
    --model_name "debug" 