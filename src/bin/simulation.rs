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
use generational_arena::Arena;
use itertools::Itertools;
use list_routine_learning_rs::*;
use polytype::atype::with_ctx;
use programinduction::{
    trs::{
        mcts::{MCTSObj, MCTSStateEvaluator, MaxThompsonMoveEvaluator, Move, TRSMCTS},
        Datum as TRSDatum, Lexicon, TRS,
    },
    MCTSManager,
};
use rand::{thread_rng, Rng};
use regex::Regex;
use std::{
    cmp::Ordering,
    f64,
    fs::File,
    io::{BufReader, Write},
    path::PathBuf,
    process::exit,
    str,
    time::Instant,
};
use term_rewriting::{trace::Trace, Operator, Rule, Term};

type Prediction = (usize, usize, String);
type Predictions = Vec<Prediction>;

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
            all_filename,
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
            .map(|e| e.to_rule(&lex, c))
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
        let mut prediction_fd = exit_err(init_out_file(&prediction_filename, true), "bad file");
        let mut best_fd = exit_err(init_out_file(&best_filename, false), "bad file");
        let mut reservoir = Reservoir::with_capacity(10000);
        for run in 0..runs {
            let mut predictions = Vec::with_capacity(data.len());
            let search_time = exit_err(
                search(
                    lex.clone(),
                    &background,
                    &data[..params.simulation.n_predictions],
                    &mut predictions,
                    &mut params.clone(),
                    &mut best_fd,
                    &mut prediction_fd,
                    &mut reservoir,
                    (&problem, run, order),
                    rng,
                ),
                "search failed",
            );
            let elapsed = start.elapsed().as_secs_f64();
            report_time(search_time, elapsed);
        }
        let mut all_fd = exit_err(init_out_file(&all_filename, false), "bad file");
        for item in reservoir.to_vec().into_iter().map(|item| item.data) {
            exit_err(str_err(writeln!(all_fd, "{}", item)), "bad file");
        }
    })
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
        lex.has_operator(Some("C"), 0)
            .or_else(|_| Err(String::from("No target concept"))),
    )
}

pub fn logsumexp(lps: &[f64]) -> f64 {
    let largest = lps.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let x = lps.iter().map(|lp| (lp - largest).exp()).sum::<f64>().ln();
    largest + x
}

fn process_prediction<'ctx, 'b>(
    mcts: &TRSMCTS<'ctx, 'b>,
    query: &Rule,
    best: &SimObj<'ctx, 'b>,
    params: &Params,
    predictions: &mut Predictions,
) -> bool {
    let n_hyps = mcts.hypotheses.len();
    let input = &query.lhs;
    let output = &query.rhs[0];
    let prediction = make_prediction(&best.trs, input, params);
    let correct = prediction == *output;
    predictions.push((correct as usize, n_hyps, best.trs.to_string()));
    correct
}

fn make_prediction<'a, 'b>(trs: &TRS<'a, 'b>, input: &Term, params: &Params) -> Term {
    let utrs = trs.full_utrs();
    let lex = trs.lexicon();
    let sig = lex.signature();
    let trace = Trace::new(
        &utrs,
        sig,
        input,
        params.model.likelihood.p_observe,
        params.model.likelihood.max_steps,
        params.model.likelihood.max_size,
        params.model.likelihood.strategy,
    );
    let best = trace
        .iter()
        .max_by(|n1, n2| {
            trace[*n1]
                .log_p()
                .partial_cmp(&trace[*n2].log_p())
                .or(Some(Ordering::Less))
                .unwrap()
        })
        .unwrap();
    trace[best].term().clone()
}

fn update_timeout(correct: bool, timeout: &mut usize, ceiling: usize, scale: f64) {
    if !correct && *timeout == ceiling {
        //notice(format!("timeout remains at ceiling of {}s", timeout), 1);
        return;
    }
    // let change = if correct { "de" } else { "in" };
    let factor = if correct { scale } else { scale.recip() };
    *timeout = ((*timeout as f64) * factor).ceil().min(ceiling as f64) as usize;
    // notice(format!("timeout {}creased to {}s", change, timeout), 1);
}

fn init_out_file(filename: &str, long: bool) -> Result<std::fs::File, String> {
    let mut fd = str_err(std::fs::File::create(filename))?;
    if long {
        str_err(writeln!(
            fd,
            "problem,run,order,trial,time,count,lmeta,ltrs,lgen,lacc,lposterior,accuracy,trs,metaprogram"
        ))?;
    } else {
        str_err(writeln!(
            fd,
            "problem,run,order,trial,time,count,lmeta,ltrs,lgen,lacc,lposterior,trs,metaprogram"
        ))?;
    }
    Ok(fd)
}

fn search<'ctx, 'b, R: Rng>(
    lex: Lexicon<'ctx, 'b>,
    background: &'b [Rule],
    data: &'b [Rule],
    predictions: &mut Predictions,
    params: &mut Params,
    best_fd: &mut std::fs::File,
    prediction_fd: &mut std::fs::File,
    reservoir: &mut Reservoir<String>,
    (problem, run, order): (&str, usize, usize),
    rng: &mut R,
) -> Result<f64, String> {
    let mut top_n = TopN::new(params.simulation.top_n);
    let mut manager = make_manager(lex, background, params, &[], rng);
    let mut timeout = params.simulation.timeout;
    let mut n_hyps;
    let mut n_step;
    let trs_data_owned = (0..data.len())
        .map(|n_data| {
            let mut cd = (0..n_data)
                .map(|idx| TRSDatum::Full(data[idx].clone()))
                .collect_vec();
            cd.push(TRSDatum::Partial(data[n_data].lhs.clone()));
            cd
        })
        .collect_vec();
    let trs_data = trs_data_owned
        .iter()
        .map(|data| data.iter().collect_vec())
        .collect_vec();
    let start = Instant::now();
    for n_data in 0..data.len() {
        let now = Instant::now();
        manager.tree_mut().mcts_mut().start_trial();
        update_data(
            &mut manager,
            &mut top_n,
            &trs_data[n_data],
            params.simulation.top_n,
            rng,
        );
        println!("\ncurrent topN:");
        top_n.iter().sorted().rev().enumerate().for_each(|(i, h)| {
            println!(
                "{},{}",
                i,
                hypothesis_string(
                    problem,
                    run,
                    order,
                    n_data + 1,
                    &h.data.trs,
                    &h.data.hyp.moves,
                    h.data.hyp.time,
                    h.data.hyp.count,
                    &[
                        h.data.hyp.obj_meta,
                        h.data.hyp.obj_trs,
                        h.data.hyp.obj_gen,
                        h.data.hyp.obj_acc,
                        h.data.hyp.ln_predict_posterior,
                    ],
                    None,
                )
            )
        });
        println!("");
        n_step = manager.step_until(rng, |_| now.elapsed().as_secs_f64() > (timeout as f64));
        manager.tree_mut().mcts_mut().finish_trial();
        let old_best = top_n.least().map(|scored| (*scored.data).clone());
        n_hyps = manager.tree().mcts().hypotheses.len();
        for (_, hyp) in manager.tree().mcts().hypotheses.iter() {
            if let Some(obj) = SimObj::try_new(hyp, manager.tree().mcts()) {
                top_n.add(ScoredItem {
                    score: -obj.hyp.ln_predict_posterior,
                    data: Box::new(obj),
                })
            } else {
                println!("FAILED: {}", hyp.count);
            }
        }
        // Make a prediction.
        let query = &data[n_data];
        let new_best = top_n.least().map(|scored| (*scored.data).clone()).unwrap();
        let correct =
            process_prediction(manager.tree().mcts(), query, &new_best, params, predictions);
        record_hypotheses(
            &manager.tree().mcts().hypotheses,
            old_best,
            new_best,
            best_fd,
            prediction_fd,
            reservoir,
            problem,
            run,
            order,
            n_data + 1,
            manager.tree().mcts(),
            correct,
            rng,
        )?;
        update_timeout(
            correct,
            &mut timeout,
            params.simulation.timeout,
            params.simulation.confidence,
        );
        println!(
            "{},{},{},{},{},{},{},{},{}",
            problem,
            run,
            order,
            n_data + 1,
            n_step,
            manager.tree().tree().tree_size(),
            n_hyps,
            manager.tree().mcts().search_time,
            start.elapsed().as_secs_f64(),
        );
    }
    // NOTE: skipping for actual runs --- a disk hog.
    // if n_data == data.len() - 1 {
    //     manager
    //         .tree()
    //         .to_file(out_file)
    //         .map_err(|_| "Record failed")?;
    // }
    Ok(manager.tree().mcts().search_time)
}

fn record_hypotheses<'ctx, 'b, R: Rng>(
    hypotheses: &Arena<Box<MCTSObj<'ctx>>>,
    old_best: Option<SimObj<'ctx, 'b>>,
    new_best: SimObj<'ctx, 'b>,
    best_fd: &mut std::fs::File,
    prediction_fd: &mut std::fs::File,
    reservoir: &mut Reservoir<String>,
    problem: &str,
    run: usize,
    order: usize,
    trial: usize,
    mcts: &TRSMCTS<'ctx, 'b>,
    correct: bool,
    rng: &mut R,
) -> Result<(), String> {
    // best
    let seed = match old_best {
        None => (None, std::f64::NEG_INFINITY),
        Some(ref h) => (None, h.hyp.ln_predict_posterior),
    };
    let best =
        hypotheses
            .iter()
            .sorted_by_key(|(_, y)| y.count)
            .fold(vec![seed], |mut acc, (i, h)| {
                let top_scorer = acc.last().expect("top_scorer");
                if h.ln_predict_posterior > top_scorer.1 {
                    acc.push((Some(i), h.ln_predict_posterior));
                }
                acc
            });
    for (opt, _) in best {
        match opt {
            None => {
                if let Some(ref hyp) = old_best {
                    str_err(record_simobj(
                        best_fd, problem, run, order, trial, hyp, None,
                    ))?;
                }
            }
            Some(i) => {
                str_err(record_hypothesis(
                    best_fd,
                    problem,
                    run,
                    order,
                    trial,
                    &hypotheses[i],
                    mcts,
                    None,
                ))?;
            }
        }
    }
    // predictions
    // TODO: FIXME: This isn't the actual hypothesis we used.
    str_err(record_simobj(
        prediction_fd,
        problem,
        run,
        order,
        trial,
        &new_best,
        Some(correct),
    ))?;
    // all
    for (_, h) in hypotheses.iter().sorted_by_key(|(_, y)| y.count) {
        reservoir.add(
            || {
                let trs = h.play(mcts).expect("trs");
                hypothesis_string(
                    problem,
                    run,
                    order,
                    trial,
                    &trs,
                    &h.moves,
                    h.time,
                    h.count,
                    &[
                        h.obj_meta,
                        h.obj_trs,
                        h.obj_gen,
                        h.obj_acc,
                        h.ln_predict_posterior,
                    ],
                    None,
                )
            },
            rng,
        );
    }
    Ok(())
}

fn record_simobj<'ctx, 'b>(
    f: &mut std::fs::File,
    problem: &str,
    run: usize,
    order: usize,
    trial: usize,
    obj: &SimObj<'ctx, 'b>,
    correct: Option<bool>,
) -> std::io::Result<()> {
    writeln!(
        f,
        "{}",
        hypothesis_string(
            problem,
            run,
            order,
            trial,
            &obj.trs,
            &obj.hyp.moves,
            obj.hyp.time,
            obj.hyp.count,
            &[
                obj.hyp.obj_meta,
                obj.hyp.obj_trs,
                obj.hyp.obj_gen,
                obj.hyp.obj_acc,
                obj.hyp.ln_predict_posterior
            ],
            correct,
        )
    )
}

fn record_hypothesis<'ctx, 'b>(
    f: &mut std::fs::File,
    problem: &str,
    run: usize,
    order: usize,
    trial: usize,
    obj: &MCTSObj<'ctx>,
    mcts: &TRSMCTS<'ctx, 'b>,
    correct: Option<bool>,
) -> std::io::Result<()> {
    writeln!(
        f,
        "{}",
        hypothesis_string(
            problem,
            run,
            order,
            trial,
            &obj.play(mcts).expect("trs"),
            &obj.moves,
            obj.time,
            obj.count,
            &[
                obj.obj_meta,
                obj.obj_trs,
                obj.obj_gen,
                obj.obj_acc,
                obj.ln_predict_posterior
            ],
            correct,
        )
    )
}

fn hypothesis_string(
    problem: &str,
    run: usize,
    order: usize,
    trial: usize,
    trs: &TRS,
    moves: &[Move],
    time: f64,
    count: usize,
    objective: &[f64],
    correct: Option<bool>,
) -> String {
    let trs_str = trs.to_string().lines().join(" ");
    let objective_string = format!("{:.4}", objective.iter().format(","));
    let meta_string = format!("{}", moves.iter().format("."));
    match correct {
        None => format!(
            "\"{}\",{},{},{},{:.9},{},{},\"{}\",\"{}\"",
            problem, run, order, trial, time, count, objective_string, trs_str, meta_string,
        ),
        Some(result) => format!(
            "\"{}\",{},{},{},{:.9},{},{},{},\"{}\",\"{}\"",
            problem, run, order, trial, time, count, objective_string, result, trs_str, meta_string,
        ),
    }
}

fn update_data<'a, 'b, R: Rng>(
    manager: &mut MCTSManager<TRSMCTS<'a, 'b>>,
    top_n: &mut TopN<Box<SimObj<'a, 'b>>>,
    data: &'b [&'b TRSDatum],
    prune_n: usize,
    rng: &mut R,
) {
    // 0. Update the data.
    manager.tree_mut().mcts_mut().data = data;

    // 1. Update the top_n.
    for mut h in std::mem::replace(top_n, TopN::new(prune_n)).to_vec() {
        h.data.update_posterior(manager.tree().mcts());
        h.score = -h.data.hyp.ln_predict_posterior;
        top_n.add(h);
    }

    // 2. Reset the MCTS store.
    manager.tree_mut().mcts_mut().clear();
    let root_state = manager.tree_mut().mcts_mut().root();

    // 3. Prune the tree store.
    let paths = top_n
        .iter()
        .sorted()
        .rev()
        .filter(|h| h.data.hyp.test_path(manager.tree().mcts()))
        .map(|h| h.data.hyp.moves.clone())
        .collect_vec();
    manager
        .tree_mut()
        .prune_except(paths.into_iter(), root_state, rng);
}

fn make_manager<'ctx, 'b, R: Rng>(
    lex: Lexicon<'ctx, 'b>,
    background: &'b [Rule],
    params: &Params,
    data: &'b [&'b TRSDatum],
    rng: &mut R,
) -> MCTSManager<TRSMCTS<'ctx, 'b>> {
    let mut mcts = TRSMCTS::new(
        lex,
        background,
        params.simulation.deterministic,
        params.simulation.lo,
        params.simulation.hi,
        data,
        params.model,
        params.mcts,
    );
    let state_eval = MCTSStateEvaluator;
    let move_eval = MaxThompsonMoveEvaluator;
    let root = mcts.root();
    MCTSManager::new(mcts, root, state_eval, move_eval, rng)
}
