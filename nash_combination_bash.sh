#!/bin/bash

# Start and end indices
START=31
END=81

# Path to your Python script
PYTHON_SCRIPT="nash_equilibrium_combination.py"

# Path to your Python executable
PYTHON_EXE="C:/Users/Lasal Jayawardena/anaconda3/envs/nlp-cw/python.exe"

# Loop from START to END
for i in $(seq $START $END); do
    echo "Running experiment with index: $i"
    "$PYTHON_EXE" "$PYTHON_SCRIPT" $i
done