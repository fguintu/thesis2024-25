#!/bin/bash
#SBATCH --job-name=pop_db              # Job name
#SBATCH --output=pop_out/output1.txt   # Standard output file
#SBATCH --error=error1.txt             # Standard error file
#SBATCH --time=1-0:00:00               # Time limit (e.g., 12 hour)
#SBATCH --ntasks=1                     # Number of tasks (1 task per job)
#SBATCH --cpus-per-task=32             # Number of CPU cores per task
#SBATCH --mem=512G                     # Memory per job (128GB)
#SBATCH --partition=day-long-cpu       # Partition to submit to (adjust as needed)
#SBATCH --mail-type=END                # Send email at job completion
#SBATCH --mail-user=frederic.guintu@emory.edu   # Email address for notifications

source /home/fredericj/skel/Desktop/myenv/bin/activate

python populate_db.py uniref90-10pm.db /data/bromberg/database/afdb_prott5_embs 0-69