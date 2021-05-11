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
        metaprogram::{MetaProgram, MetaProgramControl, MetaProgramHypothesis, Move, Temperature},
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
            _runs,
            problem_filename,
            best_filename,
            _prediction_filename,
            _all_filename,
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
        println!("problem,run,order,trial,steps,tree,hypotheses,search_time,total_time");
        let search_time = exit_err(
            search_online(
                lex.clone(),
                &background,
                &data,
                params.simulation.n_predictions,
                &mut params.clone(),
                (&problem, order),
                &best_filename,
                rng,
            ),
            "search failed",
        );
        let elapsed = start.elapsed().as_secs_f64();
        report_time(search_time, elapsed);
    })
}

fn temp(i: usize, n: usize, max_t: usize) -> f64 {
    let temp = (i as f64 * (max_t as f64).ln() / ((n - 1) as f64)).exp();
    println!("temp: {} {} {} -> {}", i, n, max_t, temp);
    temp
}

fn report_time(search_time: f64, total_time: f64) {
    notice(format!("search time: {:.3}s", search_time), 0);
    notice(format!("total time: {:.3}s", total_time), 0);
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
    train_set_size: usize,
    params: &mut Params,
    (problem, order): (&str, usize),
    best_filename: &str,
    rng: &mut R,
) -> Result<f64, String> {
    let mut best_file = std::fs::File::create(best_filename).expect("file");
    let mut top_n: TopN<Box<MetaProgramHypothesisWrapper>> = TopN::new(params.simulation.top_n);
    let timeout = params.simulation.timeout;
    // TODO: hacked in constants.
    let mpctl = MetaProgramControl::new(&[], &params.model, params.mcts.atom_weights, 7, 50);
    let mut t0 = TRS::new_unchecked(&lex, params.simulation.deterministic, background, vec![]);
    t0.set_bounds(params.simulation.lo, params.simulation.hi);
    t0.identify_symbols();
    let p0 = MetaProgram::from(t0);
    let h0 = MetaProgramHypothesis::new(mpctl, p0);
    let mut ctl = Control::new(0, 0, 0, 0, 0);
    let swap = 5000;
    //let ladder = TemperatureLadder(vec![
    //    Temperature::new(temp(0, 5, 12), temp(0, 5, 12)),
    //    Temperature::new(temp(1, 5, 12), temp(1, 5, 12)),
    //    Temperature::new(temp(2, 5, 12), temp(2, 5, 12)),
    //    Temperature::new(temp(3, 5, 12), temp(3, 5, 12)),
    //    Temperature::new(temp(4, 5, 12), temp(4, 5, 12)),
    //]);
    let ladder = TemperatureLadder(vec![Temperature::new(1.0, 1.0)]);
    let mut chain = ParallelTempering::new(h0, &[], ladder, swap, rng);
    let mut best;
    let train_data = (0..=train_set_size)
        .map(|n| convert_examples_to_data(&examples[..n]))
        .collect_vec();
    let borrowed_data = train_data
        .iter()
        .map(|x| x.iter().collect_vec())
        .collect_vec();
    for n_data in 0..=train_set_size {
        update_data_mcmc(
            &mut chain,
            &mut top_n,
            &borrowed_data[n_data],
            params.simulation.top_n,
        );
        ctl.runtime += timeout * 1000;
        best = std::f64::INFINITY;
        {
            while let Some(sample) = chain.internal_next(&mut ctl, rng) {
                print_hypothesis_mcmc(problem, order, n_data, &sample, true);
                let score = -sample.at_temperature(Temperature::new(2.0, 1.0));
                if score < best {
                    writeln!(
                        &mut best_file,
                        "{}",
                        hypothesis_string_mcmc(problem, order, n_data, &sample, true)
                    )
                    .expect("written");
                    best = score;
                }
                top_n.add(ScoredItem {
                    score,
                    data: Box::new(MetaProgramHypothesisWrapper(sample.clone())),
                });
            }
        }
        if n_data == train_set_size {
            let mut n_correct = 0;
            let mut n_tried = 0;
            let best = &top_n.least().unwrap().data.0.state;
            for query in &examples[train_set_size..] {
                let correct = process_prediction(query, &best.trs, params);
                n_correct += correct as usize;
                n_tried += 1;
            }
            println!("# END OF SEARCH");
            println!("# top hypotheses:");
            top_n.iter().sorted().enumerate().rev().for_each(|(i, h)| {
                println!(
                    "# {}\t{}",
                    i,
                    hypothesis_string_mcmc(problem, order, train_set_size, &h.data.0, true)
                )
            });
            println!("#");
            println!("# problem: {}", problem);
            println!("# order: {}", order);
            println!("# samples: {:?}", chain.samples());
            println!("# acceptance ratio: {:?}", chain.acceptance_ratio());
            println!("# swap ratio: {:?}", chain.swaps());
            println!("# best hypothesis metaprogram: {}", best.path);
            println!(
                "# best hypothesis TRS: {}",
                best.trs.to_string().lines().join(" ")
            );
            println!("# correct predictions rational: {}/{}", n_correct, n_tried);
            // TODO: fix search time
        }
    }
    Ok(0.0)
}

fn search_batch<'ctx, 'b, R: Rng>(
    lex: Lexicon<'ctx, 'b>,
    background: &'b [Rule],
    examples: &'b [Rule],
    train_set_size: usize,
    params: &mut Params,
    (problem, order): (&str, usize),
    best_filename: &str,
    rng: &mut R,
) -> Result<f64, String> {
    let now = Instant::now();
    let mut best_file = std::fs::File::create(best_filename).expect("file");
    let mut top_n: TopN<Box<MetaProgramHypothesisWrapper>> = TopN::new(params.simulation.top_n);
    // TODO: hacked in constants.
    let mpctl = MetaProgramControl::new(&[], &params.model, params.mcts.atom_weights, 7, 50);
    let timeout = params.simulation.timeout;
    let train_data = convert_examples_to_data(&examples[..train_set_size]);
    let borrowed_data = train_data.iter().collect_vec();
    let mut t0 = TRS::new_unchecked(&lex, params.simulation.deterministic, background, vec![]);
    t0.set_bounds(params.simulation.lo, params.simulation.hi);
    t0.identify_symbols();
    let p0 = MetaProgram::from(t0);
    let h0 = MetaProgramHypothesis::new(mpctl, p0);
    let mut ctl = Control::new(0, timeout * 1000, 0, 0, 0);
    // TODO: fix me
    //mcts.start_trial();
    let swap = 5000;
    let ladder = TemperatureLadder(vec![
        Temperature::new(temp(0, 5, 12), temp(0, 5, 12)),
        Temperature::new(temp(1, 5, 12), temp(1, 5, 12)),
        Temperature::new(temp(2, 5, 12), temp(2, 5, 12)),
        Temperature::new(temp(3, 5, 12), temp(3, 5, 12)),
        Temperature::new(temp(4, 5, 12), temp(4, 5, 12)),
    ]);
    let mut chain = ParallelTempering::new(h0, &borrowed_data, ladder, swap, rng);
    // TODO: should go after trial start, but am here to avoid mutability issues.
    update_data_mcmc(
        &mut chain,
        &mut top_n,
        &borrowed_data,
        params.simulation.top_n,
    );
    let mut best = std::f64::INFINITY;
    {
        println!("# drawing samples: {}ms", now.elapsed().as_millis());
        while let Some(sample) = chain.internal_next(&mut ctl, rng) {
            print_hypothesis_mcmc(problem, order, train_set_size, &sample, true);
            let score = -sample.at_temperature(Temperature::new(2.0, 1.0));
            if score < best {
                // write to best file;
                writeln!(
                    &mut best_file,
                    "{}",
                    hypothesis_string_mcmc(problem, order, train_set_size, &sample, true)
                )
                .expect("written");
                best = score;
            }
            top_n.add(ScoredItem {
                // TODO: magic constant.
                score,
                data: Box::new(MetaProgramHypothesisWrapper(sample.clone())),
            });
        }
    }
    // TODO: fix me
    // mcts.finish_trial();
    let mut n_correct = 0;
    let mut n_tried = 0;
    let best = &top_n.least().unwrap().data.0.state;
    for query in &examples[train_set_size..] {
        let correct = process_prediction(query, &best.trs, params);
        n_correct += correct as usize;
        n_tried += 1;
    }
    println!("# END OF SEARCH");
    println!("# top hypotheses:");
    top_n.iter().sorted().enumerate().rev().for_each(|(i, h)| {
        println!(
            "# {}\t{}",
            i,
            hypothesis_string_mcmc(problem, order, train_set_size, &h.data.0, true)
        )
    });
    println!("#");
    println!("# problem: {}", problem);
    println!("# order: {}", order);
    println!("# samples: {:?}", chain.samples());
    println!("# acceptance ratio: {:?}", chain.acceptance_ratio());
    println!("# swap ratio: {:?}", chain.swaps());
    println!("# best hypothesis metaprogram: {}", best.path);
    println!(
        "# best hypothesis TRS: {}",
        best.trs.to_string().lines().join(" ")
    );
    println!("# correct predictions rational: {}/{}", n_correct, n_tried);
    //println!("# correct predictions float: {}", n_correct / n_tried);
    // TODO: fix search time
    Ok(0.0)
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
    trial: usize,
    h: &MetaProgramHypothesis,
    print_trs: bool,
) {
    println!(
        "{}",
        hypothesis_string_mcmc(problem, order, trial, h, print_trs)
    );
}

fn hypothesis_string_mcmc(
    problem: &str,
    order: usize,
    trial: usize,
    h: &MetaProgramHypothesis,
    print_trs: bool,
) -> String {
    hypothesis_string_inner(
        problem,
        order,
        trial,
        h.state.trs().unwrap(),
        &h.state.metaprogram().unwrap().iter().cloned().collect_vec(),
        h.birth.time,
        h.birth.count,
        &[
            // TODO: fixme
            // h.ln_meta,
            h.ln_trs,
            // h.ln_wf,
            h.ln_acc,
            h.at_temperature(Temperature::new(2.0, 1.0)),
        ],
        None,
        print_trs,
    )
}

fn hypothesis_string_inner(
    problem: &str,
    order: usize,
    trial: usize,
    trs: &TRS,
    moves: &[Move],
    time: usize,
    count: usize,
    objective: &[f64],
    _correct: Option<bool>,
    print_trs: bool,
) -> String {
    //let trs_str = trs.to_string().lines().join(" ");
    let trs_len = trs.size();
    let objective_string = format!("{: >10.4}", objective.iter().format("\t"));
    let meta_string = format!("{}", moves.iter().format("."));
    let trs_string = format!("{}", trs).lines().join(" ");
    if print_trs {
        format!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t\"{}\"\t",
            problem, order, trial, trs_len, objective_string, count, time, trs_string
        )
    } else {
        format!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t\"{}\"\t\"{}\"",
            problem, order, trial, trs_len, objective_string, count, time, trs_string, meta_string
        )
    }
}

fn update_data_mcmc<'a, 'b>(
    chain: &mut ParallelTempering<MetaProgramHypothesis<'a, 'b>>,
    top_n: &mut TopN<Box<MetaProgramHypothesisWrapper<'a, 'b>>>,
    data: &'b [&'b TRSDatum],
    prune_n: usize,
) {
    // 0. Update the top_n.
    for mut h in std::mem::replace(top_n, TopN::new(prune_n)).to_vec() {
        h.data.0.ctl.data = data;
        h.data.0.compute_posterior(data, None);
        h.score = -h.data.0.at_temperature(Temperature::new(2.0, 1.0));
        top_n.add(h);
    }

    // 1. Update the chain.
    match top_n.least() {
        Some(best) => {
            println!(
                "BEST: {}",
                format!("{}", best.data.0.state.trs).lines().join(" ")
            );
            chain.set_data(data, true);
            for (_, thread) in chain.pool.iter_mut() {
                thread.current_mut().clone_from(&best.data.0);
                println!(
                    "UPDATE: {}",
                    format!("{}", thread.current().state.trs).lines().join(" ")
                );
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
