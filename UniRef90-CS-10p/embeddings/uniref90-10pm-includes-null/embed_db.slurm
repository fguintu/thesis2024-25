#!/bin/bash
#SBATCH --job-name=embedding_job
#SBATCH --output=embedding_output.log
#SBATCH --error=embedding_error.log
#SBATCH --partition=day-long-gpu   # Use day-long-gpu partition
#SBATCH --gres=gpu:1               # Request 1 GPU
#SBATCH --ntasks=1                 # 1 task (job step)
#SBATCH --cpus-per-task=4          # 4 CPUs to handle tokenization and database
#SBATCH --mem=32G                  # 32GB of memory for handling the database and model
#SBATCH --time=12:00:00            # Set the time limit to 12 hours (hh:mm:ss)
#SBATCH --mail-type=END                # Send email at job completion
#SBATCH --mail-user=frederic.guintu@emory.edu   # Email address for notifications

# Print out which file is being processed
echo "Processing file: sep1000.db"

# Activate the Python virtual environment
source /home/fredericj/skel/Desktop/myenv/bin/activate

# Run the python script
python prott5_db_embedder.py
