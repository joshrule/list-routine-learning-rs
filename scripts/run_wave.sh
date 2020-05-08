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

# Update a generic readme.
echo "- [\`$DIRNAME\`]($DIRNAME/readme.Rmd): $5" >> out/readme.md

# Update a simulation csv.
[ ! -e out/simulations.csv ] && echo "directory,parameter_file,problem_file,num_runs,num_jobs,reason" >> out/simulations.csv
echo "\"$DIRNAME\",\"$DIRNAME/parameters.toml\",\"$DIRNAME/problems.txt\",$3,$4,\"$5\"" >> out/simulations.csv

# Create a simulation-specific analysis.
echo "---" >> $OUTDIR/readme.Rmd
echo "title: \"$DIRNAME Analysis\"" >> $OUTDIR/readme.Rmd
echo "author: Josh Rule" >> $OUTDIR/readme.Rmd
echo "date: \"$(date +%m/%d/%Y)\"" >> $OUTDIR/readme.Rmd
echo "output:" >> $OUTDIR/readme.Rmd
echo "  html_document:" >> $OUTDIR/readme.Rmd
echo "    fig_caption: true" >> $OUTDIR/readme.Rmd
echo "    toc: true" >> $OUTDIR/readme.Rmd
echo "    toc_float:" >> $OUTDIR/readme.Rmd
echo "      collapsed: true" >> $OUTDIR/readme.Rmd
echo "---" >> $OUTDIR/readme.Rmd
echo "" >> $OUTDIR/readme.Rmd
echo "$5" >> $OUTDIR/readme.Rmd
echo "" >> $OUTDIR/readme.Rmd
echo "- [\`parameters\`](./parameters.toml)" >> $OUTDIR/readme.Rmd
echo "- [\`problems\`](./problems.txt)" >> $OUTDIR/readme.Rmd
echo "- [\`jobs\`](./job.txt)" >> $OUTDIR/readme.Rmd
echo "- [\`results\`](./results.csv)" >> $OUTDIR/readme.Rmd
echo "- number of problems: $(wc -l $OUTDIR/problems.txt| awk '{print $1;}')" >> $OUTDIR/readme.Rmd
echo "- number of runs: $3" >> $OUTDIR/readme.Rmd
echo "- number of jobs: $4" >> $OUTDIR/readme.Rmd
echo "" >> $OUTDIR/readme.Rmd
echo "```{r analysis, echo=F, warning=F, results='asis'}" >> $OUTDIR/readme.Rmd
echo "data_dir <- "1583157552_2020-03-02-13-59-12"" >> $OUTDIR/readme.Rmd
echo "res <- knitr::knit_child('../analysis.Rmd', quiet=T)" >> $OUTDIR/readme.Rmd
echo "cat(res, sep = '\n')" >> $OUTDIR/readme.Rmd
echo "```" >> $OUTDIR/readme.Rmd

# Run the tests.
parallel --joblog $OUTDIR/job.txt --jobs=$4 "RUST_BACKTRACE=full cargo run --release --bin simulation -- $1 {1} $OUTDIR/{=1 s:\/:\.:g; =}_{2}.json &> $OUTDIR/{=1 s:\/:\.:g; =}_{2}.log" ::: $(cat $2) ::: $(seq -w 0 `expr $3 - 1`)
# DEBUG
# parallel --joblog $OUTDIR/job.txt --jobs=$4 "echo RUST_BACKTRACE=full cargo run --release --bin simulation -- $1 {1} $OUTDIR/{=1 s:\/:\.:g; =}_{2}.json &> $OUTDIR/{=1 s:\/:\.:g; =}_{2}.log" ::: $(cat $2) ::: $(seq -w 0 `expr $3 - 1`)
