#!/bin/bash

# Initialize flag to 0 (no folders with more than one file found yet)
flag=0

# Loop over each folder in ./output/
for folder in ./output/*; do
    # Check if it's a directory
    if [ -d "$folder" ]; then
        # Count the number of files in the directory
        file_count=$(ls -1 "$folder" | wc -l)
        
        # Check if the folder has more than one file
        if [ "$file_count" -gt 1 ]; then
            echo "Folder $folder has more than one file."
            # Set the flag to 1 (found a folder with more than one file)
            flag=1
        fi
    fi
done

# Check if the flag is still 0 (no folders with more than one file were found)
if [ "$flag" -eq 0 ]; then
    echo "None of the folders have more than one file."
fi

