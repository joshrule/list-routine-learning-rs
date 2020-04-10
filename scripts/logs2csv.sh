#!/bin/bash

# a simple script for grabbing a bunch of log files and dumping them into a CSV.

[ $# -ne 1 ] && { echo "Usage: $0 <result_dir>"; exit 1; }

cd $1

OUTFILE="results.csv"
INFILE="job.txt"

echo "problem,run,trial,accuracy,n_seen,program" > $OUTFILE
while read -r seq host starttime jobruntime send receive exitval signal command
do
    FILE=${command##*/}
    BASENAME=${FILE%.log}
    PROBLEM=${BASENAME:0: -2}
    RUN=${BASENAME: -1}
    if [ $exitval -eq 0 ]
    then 
        tail -n 11 $FILE | sed "s/^/$PROBLEM,$RUN,/" >> $OUTFILE
    else
        echo "exited with $exitval, so SKIPPING $PROBLEM ($RUN)"
    fi
done < <(tail -n +2 $INFILE)

cd -
