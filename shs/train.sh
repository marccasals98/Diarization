#!/bin/bash
#SBATCH -D .
#SBATCH -A bsc88
#SBATCH -q acc_bscls
############# Obligatorias #######################
#SBATCH --time=2-00:00:00               # Consultar batchlim para entender los límites de las particiones. 
################# HOST ###########################
#SBATCH --nodes=1                       # Número de nodos
#SBATCH --ntasks=1                      # Número de tareas MPI totales
#SBATCH --ntasks-per-node=1             # Número de tareas MPI por nodo
#SBATCH --cpus-per-task=40              # Número de cores por tarea. Threads. $SLURM_CPUS_PER_TASK
#SBATCH --gres=gpu:1
################ Logging #########################
#SBATCH --job-name=diarization
#SBATCH --verbose
#SBATCH --output=/gpfs/projects/bsc88/speech/speaker_recognition/outputs/diarization/logs/%x_%j.txt


date +%Y-%m-%d_%H:%M:%S

# activate env
source diarization-uv/bin/activate

export TORCH_HOME="/gpfs/projects/bsc88/speech/speaker_recognition/outputs/diarization/cache/torch"

srun python src/train.py \
    --model_name "train" \
    --max_epochs 20 \
    --dc_loss_ratio 0.0