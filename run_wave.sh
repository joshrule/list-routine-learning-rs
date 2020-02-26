#!/bin/bash

[ $# -le 4 ] && { echo "Usage: $0 <prob_dir> <sim_file> <out_dir> <# runs> <# jobs>"; exit 1; }
parallel --joblog $3/log.job --jobs=$5 "RUST_BACKTRACE=full cargo run --release --bin simulation -- $2 $1/{1} &> $3/{1}_{2}.log" ::: $(find $1 -maxdepth 1 -mindepth 1 -type d -printf %f\\n) ::: $(seq -w 0 `expr $4 - 1`)
