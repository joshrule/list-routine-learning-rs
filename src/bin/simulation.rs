//! [Rust][1] simulations using input/output examples to learn [typed][2] first-order [term rewriting systems][3] that perform list routines.
//!
//! [1]: https://www.rust-lang.org
//! "The Rust Programming Language"
//! [2]: https://en.wikipedia.org/wiki/Hindley%E2%80%93Milner_type_system
//! "Wikipedia - Hindley-Milner Type System"
//! [3]: https://en.wikipedia.org/wiki/Rewriting#Term_rewriting_systems
//! "Wikipedia - Term Rewriting Systems"

use itertools::Itertools;
use list_routine_learning_rs::*;
use polytype::atype::with_ctx;
use rand::thread_rng;
use regex::Regex;
use std::{io::Write, process::exit, time::Instant};

fn main() {
    with_ctx(4096, |ctx| {
        let start = Instant::now();
        let rng = &mut thread_rng();
        let (
            params,
            runs,
            problem_filename,
            best_filename,
            prediction_filename,
            sample_filename,
            _out_filename,
        ) = exit_err(load_args(), "Failed to load parameters");
        notice("loaded parameters", 0);

        let order_regex = exit_err(str_err(Regex::new(r".+_(\d+).json")), "can't compile regex");
        let order = order_regex.captures(&problem_filename).expect("captures")[1]
            .parse::<usize>()
            .expect("order");

        let mut lex = exit_err(
            load_lexicon(&ctx, &params.simulation.signature),
            "Failed to load lexicon",
        );
        notice("loaded lexicon", 0);
        notice(&lex, 1);

        let background = exit_err(
            load_rules(&params.simulation.background, &mut lex),
            "Failed to load background",
        );
        notice("loaded background", 0);

        let c = exit_err(identify_concept(&lex), "No target concept");
        let (examples, problem) = exit_err(load_problem(&problem_filename), "Problem loading data");
        let data: Vec<_> = examples
            .iter()
            .map(|e| e.to_rule(&lex, c, params.model.likelihood.representation))
            .collect::<Result<Vec<_>, _>>()
            .unwrap_or_else(|_| {
                eprintln!("Data conversion failed.");
                exit(1);
            });
        notice("loaded data", 0);
        notice(
            examples
                .iter()
                .take(params.simulation.n_predictions)
                .map(|e| format!("{:?}", e))
                .join("\n"),
            1,
        );
        if params.simulation.n_predictions < examples.len() {
            println!("#");
            notice(
                examples
                    .iter()
                    .skip(params.simulation.n_predictions)
                    .map(|e| format!("{:?}", e))
                    .join("\n"),
                1,
            );
        }

        let mut best_file = exit_err(init_csv_fd(
            &best_filename,
            "problem\torder\trun\ttrial\tsize\tlmeta\tltrs\tlgen\tlacc\tlposterior\ttime\tcount\ttrs\tmetaprogram",
        ), "failed to open best");
        let mut prediction_file = exit_err(init_csv_fd(
            &prediction_filename,
            "problem\torder\trun\ttrial\tsize\tlmeta\tltrs\tlgen\tlacc\tlposterior\ttime\tcount\taccuracy\ttrs\tmetaprogram",
        ), "failed to open predictions");
        let mut sample_file = exit_err(init_csv_fd(
            &sample_filename,
            "problem\torder\trun\ttrial\tsize\tlmeta\tltrs\tlgen\tlacc\tlposterior\ttime\tcount\ttrs\tmetaprogram",
        ), "failed to open best");
        let mut reservoir = Reservoir::with_capacity(10000);
        for run in 0..runs {
            println!("#");
            notice("beginning search", 0);
            exit_err(
                match params.simulation.mode {
                    SimulationMode::Online => search_online(
                        lex.clone(),
                        &background,
                        &data,
                        params.simulation.n_predictions,
                        &mut params.clone(),
                        (&problem, order, run),
                        &mut best_file,
                        &mut prediction_file,
                        &mut reservoir,
                        rng,
                    ),
                    SimulationMode::Batch => search_batch(
                        lex.clone(),
                        &background,
                        &data,
                        params.simulation.n_predictions,
                        &mut params.clone(),
                        (&problem, order, run),
                        &mut best_file,
                        &mut prediction_file,
                        &mut reservoir,
                        rng,
                    ),
                },
                "search failed",
            );
        }
        for item in reservoir.to_vec().into_iter().map(|item| item.data) {
            exit_err(str_err(writeln!(sample_file, "{}", item)), "bad file");
        }
        let total_time = start.elapsed().as_secs_f64();
        println!("#");
        notice(format!("total time: {:.3}s", total_time), 0);
    })
}
