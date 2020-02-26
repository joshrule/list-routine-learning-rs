#!/bin/bash

# This is a script for running simulations.

# Give it a parameter file, problem file, number of runs, and number of jobs:
# - parameter_file is a TOML file containing all the parameters of the simulation.
# - problem_file is a plain text list of the directoriescontaining each problem to be run.
# - num_runs is an integer, the number of runs to complete for each problem.
# - num_jobs is an integer, the number of jobs to run in parallel.
# - description is a quoted string. What test are you running and why?
[ $# -ne 5 ] && { echo "Usage: $0 <simulation_file> <problem_file> <num_runs> <num_jobs> <description>"; exit 1; }

# Create a timestamped directory.
DIRNAME=`date "+%s_%Y-%m-%d-%H-%M-%S"`
OUTDIR=out/$DIRNAME
mkdir $OUTDIR

# Copy the parameter file and problem file into the directory.
cp $1 $OUTDIR/parameters.toml
cp $2 $OUTDIR/problems.txt

# Update a generic readme and test-specific readme and csv file.
echo "- [\`$DIRNAME\`]($DIRNAME/readme.md): $5" >> out/readme.md

echo "# \`$DIRNAME\`" >> $OUTDIR/readme.md
echo "" >> $OUTDIR/readme.md
echo "$5" >> $OUTDIR/readme.md
echo "- [\`parameter file\`](./parameters.toml)" >> $OUTDIR/readme.md
echo "- [\`problem file\`](./problems.txt)" >> $OUTDIR/readme.md
echo "- number of runs: $3" >> $OUTDIR/readme.md
echo "- number of jobs: $4" >> $OUTDIR/readme.md

[ ! -e out/simulations.csv ] && echo "directory,parameter_file,problem_file,num_runs,num_jobs,reason" >> out/simulations.csv
echo "\"$DIRNAME\",\"$DIRNAME/parameters.toml\",\"$DIRNAME/problems.txt\",$3,$4,\"$5\"" >> out/simulations.csv

# Run the tests.
parallel --joblog $OUTDIR/job.txt --jobs=$4 "RUST_BACKTRACE=full cargo run --release --bin simulation -- $1 {1} &> $OUTDIR/{=1 s:\/:\.:g; =}_{2}.log" ::: $(cat $2) ::: $(seq -w 0 `expr $3 - 1`)
