#!/bin/bash

HERE=`pwd`
parallel --progress --bar "RUST_BACKTRACE=full cargo run --release --bin simulation trs/waves/$1/{1}/simulation.toml &> $2/{1}_{2}.log" ::: $(cd trs/waves/$1 && find -maxdepth 1 -mindepth 1 -type d -printf %f\\n && cd $HERE) ::: $(seq -w 0 `expr $3 - 1`)
