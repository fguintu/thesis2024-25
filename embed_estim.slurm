#!/bin/bash
#SBATCH --job-name=emb_est             # Job name
#SBATCH --output=output.txt           s # Standard output file
#SBATCH --error=error.txt              # Standard error file
#SBATCH --time=01:00:00                # Time limit (e.g., 1 hour)
#SBATCH --ntasks=1                     # Number of tasks (1 task per job)
#SBATCH --cpus-per-task=4              # Number of CPU cores per task
#SBATCH --mem=16G                      # Memory per job (16GB)
#SBATCH --partition=short-cpu          # Partition to submit to (adjust as needed)
#SBATCH --mail-type=END                # Send email at job completion
#SBATCH --mail-user=frederic.guintu@emory.edu   # Email address for notifications

# Print out which file is being processed
echo "Processing file: batches/batch_0.fasta"
echo "Output file will be: test.h5"

# Activate the Python virtual environment
source /home/fredericj/skel/Desktop/myenv/bin/activate

# Run the python script
python prott5_embedder.py --input batches/batch_0.fasta --output test.h5 --per_protein 1
