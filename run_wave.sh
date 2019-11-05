#!/bin/bash

find trs/waves/$1 -maxdepth 1 -mindepth 1 -type d | parallel --progress --bar "RUST_BACKTRACE=1 cargo run --release --bin simulation {}/simulation.toml &> {}/`date '+%Y-%m-%d-%H-%M-%S'`.log"
