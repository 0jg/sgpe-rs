#!/bin/bash
#SBATCH --job-name=sgpe
#SBATCH --output=./slurm_out/spgpe_%A_%a.out
#SBATCH --error=./slurm_out/spgpe_%A_%a.err
#SBATCH --array=0-2047  # Change this to match the number of lines in your file
#SBATCH --mem=128G
#SBATCH --mail-user=pzpw65@dur.ac.uk

#SBATCH -p multi

# Read the values from ./mu_T_2048_uniform.txt based on SLURM_ARRAY_TASK_ID
LINE=$(sed -n "$(($SLURM_ARRAY_TASK_ID + 1))p" ./mu_T_2048_uniform.txt)

# Split the line into CHEM_POT and TEMP (assuming the values are separated by a comma)
IFS=',' read -ra VALUES <<< "$LINE"
CHEM_POT="${VALUES[0]}"
TEMP="${VALUES[1]}"

# Run the program with the values read from the file
./target/release/spgpe "$TEMP" "$CHEM_POT"
