#!/bin/bash
#SBATCH --job-name=distance_calculations              # Job name
#SBATCH --output=output.txt                           # Standard output file
#SBATCH --error=error.txt                             # Standard error file
#SBATCH --time=2-0:00:00                              # Time limit (e.g., 12 hour)
#SBATCH --ntasks=1                                    # Number of tasks (1 task per job)
#SBATCH --cpus-per-task=58                            # Number of CPU cores per task
#SBATCH --mem=116G                                    # Memory per job
#SBATCH --partition=week-long-cpu                     # Partition to submit to (adjust as needed)
#SBATCH --mail-type=END                               # Send email at job completion
#SBATCH --mail-user=frederic.guintu@emory.edu         # Email address for notifications

source /home/fredericj/skel/Desktop/myenv/bin/activate

python table_pairs.py