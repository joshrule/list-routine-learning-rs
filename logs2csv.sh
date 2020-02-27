#!/bin/bash

# a simple script for grabbing a bunch of log files and dumping them into a CSV.
cd $1

OUTFILE="results.csv"

echo "problem,run,trial,n_seen,accuracy,input,correct,predicted" > $OUTFILE
for FILE in `ls -1 *log`
do
    BASENAME=${FILE:4: -4};
    PROBLEM=${BASENAME:0: -2}
    RUN=${BASENAME: -1}
    echo "$BASENAME = $PROBLEM ($RUN)"
    tail -n 13 $FILE | head -n 11 | sed "s/^/$PROBLEM,$RUN,/" >> $OUTFILE
done

cd -
