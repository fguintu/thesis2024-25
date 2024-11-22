#!/bin/bash
#SBATCH --job-name=protein_similarity           # Job name
#SBATCH --output=protein_similarity.out         # Output file for logs
#SBATCH --error=protein_similarity.err          # Error file for logs
#SBATCH --time=24:00:00                         # Time limit (hh:mm:ss)
#SBATCH --gres=gpu:1                            # Request 1 GPU
#SBATCH --cpus-per-task=4                       # Number of CPU cores per task
#SBATCH --mem=128G                              # Memory per job (128GB)
#SBATCH --partition=day-long-gpu                # Partition to submit to (adjust as needed)
#SBATCH --mail-type=END                         # Send email at job completion
#SBATCH --mail-user=frederic.guintu@emory.edu   # Email address for notifications

# Activate the Python virtual environment
source /home/fredericj/skel/Desktop/myenv/bin/activate

# Run the python script
python network_gen.py

echo "Job ended at $(date)"
