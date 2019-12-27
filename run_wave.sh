#!/bin/bash

# $1 = wave_dir
# $2 = out_dir
# $3 = # of runs

HERE=`pwd`
parallel --joblog $2/log.job --jobs=34 "RUST_BACKTRACE=full cargo run --release --bin simulation trs/waves/$1/{1}/simulation.toml &> $2/{1}_{2}.log" ::: $(cd trs/waves/$1 && find -maxdepth 1 -mindepth 1 -type d -printf %f\\n && cd $HERE) ::: $(seq -w 0 `expr $3 - 1`)
