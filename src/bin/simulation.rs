//! [Rust][1] simulations using input/output examples to learn [typed][2] first-order [term rewriting systems][3] that perform list routines.
//!
//! [1]: https://www.rust-lang.org
//! "The Rust Programming Language"
//! [2]: https://en.wikipedia.org/wiki/Hindley%E2%80%93Milner_type_system
//! "Wikipedia - Hindley-Milner Type System"
//! [3]: https://en.wikipedia.org/wiki/Rewriting#Term_rewriting_systems
//! "Wikipedia - Term Rewriting Systems"
// TODO: Run *either* GP or MCTS.

use docopt::Docopt;
use itertools::Itertools;
use list_routine_learning_rs::*;
use polytype::atype::with_ctx;
use programinduction::{
    hypotheses::{Bayesable, Temperable},
    inference::{Control, ParallelTempering, TemperatureLadder},
    trs::{
        metaprogram::{
            LearningMode, MetaProgram, MetaProgramControl, MetaProgramHypothesis, Move, Temperature,
        },
        Datum as TRSDatum, Lexicon, TRS,
    },
};
use rand::{thread_rng, Rng};
use regex::Regex;
use std::{
    f64,
    fs::File,
    io::{BufReader, Write},
    path::PathBuf,
    process::exit,
    str,
    time::Instant,
};
use term_rewriting::{Operator, Rule};

#[derive(Clone, PartialEq, Eq)]
pub struct MetaProgramHypothesisWrapper<'ctx, 'b>(MetaProgramHypothesis<'ctx, 'b>);

impl<'ctx, 'b> Keyed for MetaProgramHypothesisWrapper<'ctx, 'b> {
    type Key = MetaProgram<'ctx, 'b>;
    fn key(&self) -> &Self::Key {
        &self.0.state.path
    }
}

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

        notice("searching", 0);
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
            exit_err(
                search_online(
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
                "search failed",
            );
        }
        for item in reservoir.to_vec().into_iter().map(|item| item.data) {
            exit_err(str_err(writeln!(sample_file, "{}", item)), "bad file");
        }
        let total_time = start.elapsed().as_secs_f64();
        notice(format!("total time: {:.3}s", total_time), 0);
    })
}

fn temp(i: usize, n: usize, max_t: usize) -> f64 {
    (i as f64 * (max_t as f64).ln() / ((n - 1) as f64)).exp()
}

fn load_args() -> Result<(Params, usize, String, String, String, String, String), String> {
    let args: Args =
        Docopt::new("Usage: sim <params> <run> <data> <best> <prediction> <all> <out>")
            .and_then(|d| d.deserialize())
            .unwrap_or_else(|e| e.exit());
    let toml_string = path_to_string(".", &args.arg_params)?;
    str_err(toml::from_str(&toml_string).map(|toml| {
        (
            toml,
            args.arg_run,
            args.arg_data.clone(),
            args.arg_best.clone(),
            args.arg_prediction.clone(),
            args.arg_all.clone(),
            args.arg_out.clone(),
        )
    }))
}

fn load_problem(data_filename: &str) -> Result<(Vec<Datum>, String), String> {
    let path: PathBuf = PathBuf::from(data_filename);
    let file = str_err(File::open(path))?;
    let reader = BufReader::new(file);
    let problem: Problem = str_err(serde_json::from_reader(reader))?;
    Ok((problem.data, problem.id))
}

fn identify_concept(lex: &Lexicon) -> Result<Operator, String> {
    str_err(
        lex.has_operator(Some("C"), 1)
            .or_else(|_| Err(String::from("No target concept"))),
    )
}

pub fn logsumexp(lps: &[f64]) -> f64 {
    let largest = lps.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let x = lps.iter().map(|lp| (lp - largest).exp()).sum::<f64>().ln();
    largest + x
}

fn search_online<'ctx, 'b, R: Rng>(
    lex: Lexicon<'ctx, 'b>,
    background: &'b [Rule],
    examples: &'b [Rule],
    n_trials: usize,
    params: &mut Params,
    (problem, order, run): (&str, usize, usize),
    best_file: &mut File,
    prediction_file: &mut File,
    reservoir: &mut Reservoir<String>,
    rng: &mut R,
) -> Result<f64, String> {
    let now = Instant::now();
    let mut search_time = 0.0;
    let mut top_n: TopN<Box<MetaProgramHypothesisWrapper>> = TopN::new(params.simulation.top_n);
    let timeout = params.simulation.timeout;
    // TODO: hacked in constants.
    let mpctl1 = MetaProgramControl::new(
        &[],
        &params.model,
        LearningMode::Refactor,
        params.mcts.atom_weights,
        7,
        50,
    );
    let mpctl2 = MetaProgramControl::new(
        &[],
        &params.model,
        LearningMode::Sample,
        params.mcts.atom_weights,
        2,
        22,
    );
    let mut t0 = TRS::new_unchecked(&lex, params.simulation.deterministic, background, vec![]);
    t0.set_bounds(params.simulation.lo, params.simulation.hi);
    t0.identify_symbols();
    let p0 = MetaProgram::from(t0);
    let h01 = MetaProgramHypothesis::new(mpctl1, &p0);
    let h02 = MetaProgramHypothesis::new(mpctl2, &p0);
    let mut ctl1 = Control::new(0, timeout * 1000, 0, 0, 0);
    let mut ctl2 = Control::new(0, timeout * 1000, 0, 0, 0);
    let swap = 5000;
    //let ladder = TemperatureLadder(vec![Temperature::new(2.0, 1.0)]);
    let ladder = TemperatureLadder(vec![
        Temperature::new(temp(0, 5, 1), temp(0, 5, 1)),
        Temperature::new(temp(1, 5, 1), temp(1, 5, 1)),
        Temperature::new(temp(2, 5, 1), temp(2, 5, 1)),
        Temperature::new(temp(3, 5, 1), temp(3, 5, 1)),
        Temperature::new(temp(4, 5, 1), temp(4, 5, 1)),
    ]);
    let mut chain1 = ParallelTempering::new(h01, &[], ladder.clone(), swap, rng);
    let mut chain2 = ParallelTempering::new(h02, &[], ladder, swap, rng);
    let mut best;
    let data = (1..=n_trials)
        .map(|n| convert_examples_to_data(&examples[..n]))
        .collect_vec();
    let borrowed_data = data.iter().map(|x| x.iter().collect_vec()).collect_vec();
    for n_data in 0..n_trials {
        update_data_mcmc(
            &mut chain1,
            &mut top_n,
            LearningMode::Refactor,
            &borrowed_data[n_data],
            params.simulation.top_n,
        );
        update_data_mcmc(
            &mut chain2,
            &mut top_n,
            LearningMode::Sample,
            &borrowed_data[n_data],
            params.simulation.top_n,
        );
        for (i, chain) in chain1.pool.iter_mut().enumerate() {
            chain.1.set_temperature(Temperature::new(
                1.0,
                temp(i, 5, n_data + 1),
            ));
        }
        for (i, chain) in chain2.pool.iter_mut().enumerate() {
            chain.1.set_temperature(Temperature::new(
                1.0,
                temp(i, 5, n_data + 1),
            ));
        }
        ctl1.runtime += timeout * 1000;
        ctl2.runtime += timeout * 1000;
        best = std::f64::INFINITY;
        let trial_start = Instant::now();
        while let (Some(sample1), Some(sample2)) = (
            chain1.internal_next(&mut ctl1, rng),
            chain2.internal_next(&mut ctl2, rng),
        ) {
            if params.simulation.verbose {
                print_hypothesis_mcmc(problem, order, run, n_data, &sample1, true, None);
            }
            let score = -sample1.at_temperature(Temperature::new(4.0, 1.0));
            if score < best {
                writeln!(
                    best_file,
                    "{}",
                    hypothesis_string_mcmc(problem, order, run, n_data, &sample1, true, None)
                )
                .expect("written");
                best = score;
            }
            top_n.add(ScoredItem {
                score,
                data: Box::new(MetaProgramHypothesisWrapper(sample1.clone())),
            });
            reservoir.add(
                || hypothesis_string_mcmc(problem, order, run, n_data, &sample1, true, None),
                rng,
            );
            if params.simulation.verbose {
                print_hypothesis_mcmc(problem, order, run, n_data, &sample2, true, None);
            }
            let score = -sample2.at_temperature(Temperature::new(4.0, 1.0));
            if score < best {
                writeln!(
                    best_file,
                    "{}",
                    hypothesis_string_mcmc(problem, order, run, n_data, &sample2, true, None)
                )
                .expect("written");
                best = score;
            }
            top_n.add(ScoredItem {
                score,
                data: Box::new(MetaProgramHypothesisWrapper(sample2.clone())),
            });
            reservoir.add(
                || hypothesis_string_mcmc(problem, order, run, n_data, &sample2, true, None),
                rng,
            );
        }
        search_time += trial_start.elapsed().as_secs_f64();
        let h_best = &top_n.least().unwrap().data.0;
        let query = &examples[n_data];
        let correct = process_prediction(query, &h_best.state.trs, params);
        writeln!(
            prediction_file,
            "{}",
            hypothesis_string_mcmc(problem, order, run, n_data, &h_best, true, Some(correct))
        )
        .ok();
        if n_data + 1 == n_trials {
            let best = &top_n.least().unwrap().data.0.state;
            println!("# END OF SEARCH");
            println!("# top hypotheses:");
            top_n.iter().sorted().enumerate().rev().for_each(|(i, h)| {
                println!(
                    "# {}\t{}",
                    i,
                    hypothesis_string_mcmc(problem, order, run, n_data, &h.data.0, true, None)
                )
            });
            println!("#");
            println!("# problem: {}", problem);
            println!("# order: {}", order);
            println!("# samples 1: {:?}", chain1.samples());
            println!("# samples 2: {:?}", chain2.samples());
            println!("# acceptance ratio 1: {:?}", chain1.acceptance_ratio());
            println!("# acceptance ratio 2: {:?}", chain2.acceptance_ratio());
            println!("# swap ratio 1: {:?}", chain1.swaps());
            println!("# swap ratio 2: {:?}", chain2.swaps());
            println!("# best hypothesis metaprogram: {}", best.path);
            println!(
                "# best hypothesis TRS: {}",
                best.trs.to_string().lines().join(" ")
            );
            println!("# search time (s): {}", search_time);
            println!("# run time (s): {}", now.elapsed().as_secs_f64());
        }
    }
    Ok(search_time)
}

fn convert_examples_to_data(examples: &[Rule]) -> Vec<TRSDatum> {
    examples
        .iter()
        .cloned()
        .enumerate()
        .map(|(i, e)| {
            if i < examples.len() - 1 {
                TRSDatum::Full(e)
            } else {
                TRSDatum::Partial(e.lhs)
            }
        })
        .collect_vec()
}

fn print_hypothesis_mcmc(
    problem: &str,
    order: usize,
    run: usize,
    trial: usize,
    h: &MetaProgramHypothesis,
    print_meta: bool,
    correct: Option<bool>,
) {
    println!(
        "{}",
        hypothesis_string_mcmc(problem, order, run, trial, h, print_meta, correct)
    );
}

fn hypothesis_string_mcmc(
    problem: &str,
    order: usize,
    run: usize,
    trial: usize,
    h: &MetaProgramHypothesis,
    print_meta: bool,
    correct: Option<bool>,
) -> String {
    hypothesis_string_inner(
        problem,
        order,
        run,
        trial,
        h.state.trs().unwrap(),
        &h.state.metaprogram().unwrap().iter().cloned().collect_vec(),
        h.birth.time,
        h.birth.count,
        &[
            h.ln_meta,
            h.ln_trs,
            h.ln_wf,
            h.ln_acc,
            h.at_temperature(Temperature::new(4.0, 1.0)),
        ],
        correct,
        print_meta,
    )
}

fn hypothesis_string_inner(
    problem: &str,
    order: usize,
    run: usize,
    trial: usize,
    trs: &TRS,
    moves: &[Move],
    time: usize,
    count: usize,
    objective: &[f64],
    correct: Option<bool>,
    print_meta: bool,
) -> String {
    //let trs_str = trs.to_string().lines().join(" ");
    let trs_len = trs.size();
    let objective_string = format!("{: >10.4}", objective.iter().format("\t"));
    let meta_string = format!("{}", moves.iter().format("."));
    let trs_string = format!("{}", trs).lines().join(" ");
    match (print_meta, correct) {
        (false, None) => {
            format!(
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t\"{}\"",
                problem, order, run, trial, trs_len, objective_string, count, time, trs_string
            )
        }
        (false, Some(c)) => {
            format!(
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t\"{}\"",
                problem, order, run, trial, trs_len, objective_string, count, time, c, trs_string
            )
        }
        (true, None) => {
            format!(
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t\"{}\"\t\"{}\"",
                problem,
                order,
                run,
                trial,
                trs_len,
                objective_string,
                count,
                time,
                trs_string,
                meta_string
            )
        }
        (true, Some(c)) => {
            format!(
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t\"{}\"\t\"{}\"",
                problem,
                order,
                run,
                trial,
                trs_len,
                objective_string,
                count,
                time,
                c,
                trs_string,
                meta_string
            )
        }
    }
}

fn update_data_mcmc<'a, 'b>(
    chain: &mut ParallelTempering<MetaProgramHypothesis<'a, 'b>>,
    top_n: &mut TopN<Box<MetaProgramHypothesisWrapper<'a, 'b>>>,
    mode: LearningMode,
    data: &'b [&'b TRSDatum],
    prune_n: usize,
) {
    // 0. Update the top_n.
    for mut h in std::mem::replace(top_n, TopN::new(prune_n)).to_vec() {
        h.data.0.ctl.data = data;
        h.data.0.compute_posterior(data, None);
        h.score = -h.data.0.at_temperature(Temperature::new(4.0, 1.0));
        top_n.add(h);
    }

    // 1. Update the chain.
    let best = top_n
        .iter()
        .filter(|x| x.data.0.ctl.mode == mode)
        .min_by(|a, b| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    match best {
        Some(best) => {
            chain.set_data(data, true);
            for (_, thread) in chain.pool.iter_mut() {
                thread.current_mut().clone_from(&best.data.0);
            }
        }
        None => {
            chain.set_data(data, true);
            for (_, thread) in chain.pool.iter_mut() {
                thread.current_mut().ctl.data = data;
            }
        }
    }
}
