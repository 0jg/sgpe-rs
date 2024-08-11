#!/bin/zsh

# Path to the mu_T_pairs.txt file
MU_T_FILE="./mu_T_3000_uniform.txt"

# Path to your compiled Rust program
PROGRAM_PATH="/Users/jack/Library/Mobile Documents/com~apple~CloudDocs/Research/temperature_model/spgpe-rs/target/release/spgpe"

# Maximum number of concurrent jobs
MAX_JOBS=5

# Array to store PIDs of active jobs
active_jobs=()

# Read each line from the mu_T_pairs file and run simulations
while IFS=, read -r mu t; do
	# Clean up finished jobs from the active jobs list
	for pid in "${active_jobs[@]}"; do
		if ! kill -0 $pid 2> /dev/null; then
			# Remove PID frothe active jobs list
			active_jobs=("${(@)active_jobs:#$pid}")
		fi
	done

	# Wait for a slot to open if necessary
	while [ ${#active_jobs[@]} -ge $MAX_JOBS ]; do
		sleep 1
		for pid in "${active_jobs[@]}"; do
			if ! kill -0 $pid 2> /dev/null; then
				# Remove PID from the active jobs list
				active_jobs=("${(@)active_jobs:#$pid}")
				break
			fi
		done
	done

	# Start a new simulation
	$PROGRAM_PATH $mu $t false &
	active_jobs+=($!)
done < "$MU_T_FILE"

# Wait for all remaining simulations to complete
for pid in $active_jobs; do
	wait $pid
done
echo "All simulations completed."
