#!/bin/sh

# Set the maximum number of attempts
max_attempts=25

# Set a counter for the number of attempts
attempt_num=1

# Set a flag to indicate whether the command was successful
success=false

# Loop until the command is successful or the maximum number of attempts is reached
while [ $success = false ] && [ $attempt_num -le $max_attempts ]; do
  # Execute the command
  /bin/sh -ec 'docker run --rm --init --ulimit core=0 -p 8070:8070 lfoppiano/grobid:0.8.0 &'
  sleep 30
  /bin/sh -ec 'python3 data_utils/parse_pdf/dir_parse.py --dir data/venue_data/aacl/2023/pos'


  # Check the exit code of the command
  if [ $? -eq 0 ]; then
    # The command was successful
    success=true
  else
    # The command was not successful
    echo "Attempt $attempt_num failed. Trying again..."
    # Increment the attempt counter
    attempt_num=$(( attempt_num + 1 ))
  fi
done

# Check if the command was successful
if [ $success = true ]; then
  # The command was successful
  echo "The command was successful after $attempt_num attempts."
  lsof -i tcp:8070 | awk 'NR!=1 {print $2}' | xargs kill
else
  # The command was not successful
  echo "The command failed after $max_attempts attempts."
fi