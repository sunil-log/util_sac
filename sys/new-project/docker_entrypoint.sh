#!/bin/bash


# tmp dir for matplotlib
export MPLCONFIGDIR="/tmp/matplotlib-$USER"
mkdir -p "$MPLCONFIGDIR"


export PYTHONPATH="/sac/src:$PYTHONPATH"


cd /sac/src

# ---------------------
# run scripts
# ---------------------



python run_main.py
# python run.py
exit