#!/bin/bash
#SBATCH --job-name=rand_gen                     # Job name
#SBATCH --output=rand_gen-output.log            # Standard output file
#SBATCH --error=rand_gen-error.log              # Standard error file
#SBATCH --time=02:00:00                         # Time limit (e.g., 1 hour)
#SBATCH --ntasks=1                              # Number of tasks (1 task per job)
#SBATCH --cpus-per-task=4                       # Number of CPU cores per task
#SBATCH --mem=16G                               # Memory per job (16GB)
#SBATCH --partition=day-long-cpu                   # Partition to submit to (adjust as needed)
#SBATCH --mail-type=END                         # Send email at job completion
#SBATCH --mail-user=frederic.guintu@emory.edu   # Email address for notifications

# Print out which file is being processed
echo "Output files will be: embeddings.npy, sequences.txt"

# Activate the Python virtual environment
source /home/fredericj/skel/Desktop/myenv/bin/activate

# Run the python script
python rgseq.py