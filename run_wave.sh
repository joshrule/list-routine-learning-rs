#!/bin/bash

find trs/waves/$1 -type d -depth 1 | parallel --progress --bar "cargo run --release --bin simulation {}/simulation.toml > {}/`date '+%Y-%m-%d-%H-%M-%S'`.log"
