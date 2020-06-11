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
    trs::{
        mcts::{MCTSModel, MCTSObj, MCTSStateEvaluator, ThompsonMoveEvaluator, TRSMCTS},
        Datum as TRSDatum, Hypothesis, Lexicon, TRS,
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
type Hyp<'a, 'b> = Hypothesis<MCTSObj<'a, 'b>, &'b TRSDatum, MCTSModel<'a, 'b>>;

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
        notice(
            format!("problem,run,order,trial,steps,seen,search_time,total_time"),
            1,
        );
        let mut prediction_fd = exit_err(init_out_file(&prediction_filename), "bad file");
        let mut best_fd = exit_err(init_out_file(&best_filename), "bad file");
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
        let mut all_fd = exit_err(init_out_file(&all_filename), "bad file");
        for item in reservoir.to_vec().into_iter().map(|item| item.data) {
            exit_err(str_err(writeln!(all_fd, "{}", item)), "bad file");
        }
    })
}

fn report_time(search_time: f64, total_time: f64) {
    notice(format!("search time: {:.3e}s", search_time), 0);
    notice(format!("total time: {:.3e}s", total_time), 0);
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

#[allow(clippy::too_many_arguments)]
fn process_prediction(
    data: &[TRSDatum],
    seen: &[Hyp],
    params: &Params,
    predictions: &mut Predictions,
) -> bool {
    let n_data = data.len() - 1;
    let n_seen = seen.len();
    if let TRSDatum::Full(ref rule) = &data[n_data] {
        let input = &rule.lhs;
        let output = &rule.rhs[0];
        let trs = best_so_far(seen);
        let prediction = make_prediction(trs, input, params);
        let correct = prediction == *output;
        predictions.push((correct as usize, n_seen, trs.to_string()));
        correct
    } else {
        panic!("passed in partial as final datum");
    }
}

fn best_so_far_pair<'a, 'b, 'c>(pop: &'c [Hyp<'a, 'b>]) -> (usize, &'c Hyp<'a, 'b>) {
    pop.iter()
        .enumerate()
        .max_by(|(_, x), (_, y)| {
            x.lposterior
                .partial_cmp(&y.lposterior)
                .or_else(|| Some(Ordering::Less))
                .unwrap()
        })
        .unwrap()
}

fn best_so_far<'a, 'b, 'c>(pop: &'c [Hyp<'a, 'b>]) -> &'c TRS<'a, 'b> {
    &best_so_far_pair(pop).1.object.trs
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

fn init_out_file(filename: &str) -> Result<std::fs::File, String> {
    let mut fd = str_err(std::fs::File::create(filename))?;
    str_err(writeln!(fd, "problem,run,order,trial,time,count,trs"))?;
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
    let mut manager = make_manager(lex, background, params, &[], rng);
    let mut timeout = params.simulation.timeout;
    let mut n_seen = 0;
    let mut n_step = 0;
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
        update_data(&mut manager, &trs_data[n_data], rng);
        let now = Instant::now();
        manager.tree_mut().mcts_mut().start_trial();
        n_step += manager.step_until(rng, |_| now.elapsed().as_secs_f64() > (timeout as f64));
        manager.tree_mut().mcts_mut().finish_trial();
        record_hypotheses(
            &manager.tree().mcts().hypotheses,
            n_seen,
            best_fd,
            prediction_fd,
            reservoir,
            problem,
            run,
            order,
            n_data + 1,
            rng,
        )?;
        n_seen = manager.tree().mcts().hypotheses.len();
        // // Make a prediction.
        let trss = &manager.tree().mcts().hypotheses;
        let prediction_data = (0..=n_data)
            .map(|idx| TRSDatum::Full(data[idx].clone()))
            .collect_vec();
        let correct = process_prediction(&prediction_data, trss, params, predictions);
        update_timeout(
            correct,
            &mut timeout,
            params.simulation.timeout,
            params.simulation.confidence,
        );
        notice(
            format!(
                "{},{},{},{},{},{},{},{}",
                problem,
                run,
                order,
                n_data + 1,
                n_step,
                n_seen,
                manager.tree().mcts().search_time,
                start.elapsed().as_secs_f64(),
            ),
            1,
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

fn record_hypotheses<R: Rng>(
    hypotheses: &[Hyp],
    n: usize,
    best_fd: &mut std::fs::File,
    prediction_fd: &mut std::fs::File,
    reservoir: &mut Reservoir<String>,
    problem: &str,
    run: usize,
    order: usize,
    trial: usize,
    rng: &mut R,
) -> Result<(), String> {
    // best
    let i = if n == 0 {
        0
    } else {
        best_so_far_pair(&hypotheses[..n]).0
    };
    let best = hypotheses
        .iter()
        .enumerate()
        .skip(n)
        .fold(vec![i], |mut acc, (i, h)| {
            let top_scorer = acc.last().expect("top_score");
            if hypotheses[*top_scorer].lposterior < h.lposterior {
                acc.push(i);
            }
            acc
        });
    for &i in &best {
        str_err(record_hypothesis(
            best_fd,
            problem,
            run,
            order,
            trial,
            &hypotheses[i].object.trs,
            hypotheses[i].object.time,
            i,
        ))?;
    }
    // predictions
    let i = best_so_far_pair(hypotheses).0;
    str_err(record_hypothesis(
        prediction_fd,
        problem,
        run,
        order,
        trial,
        &hypotheses[i].object.trs,
        hypotheses[i].object.time,
        i,
    ))?;
    // all
    for (i, h) in hypotheses.iter().enumerate().skip(n) {
        let h_str = hypothesis_string(problem, run, order, trial, &h.object.trs, h.object.time, i);
        reservoir.add(ReservoirItem::new(h_str, rng));
    }
    Ok(())
}

fn record_hypothesis(
    f: &mut std::fs::File,
    problem: &str,
    run: usize,
    order: usize,
    trial: usize,
    trs: &TRS,
    time: f64,
    count: usize,
) -> std::io::Result<()> {
    writeln!(
        f,
        "{}",
        hypothesis_string(problem, run, order, trial, trs, time, count)
    )
}

fn hypothesis_string(
    problem: &str,
    run: usize,
    order: usize,
    trial: usize,
    trs: &TRS,
    time: f64,
    count: usize,
) -> String {
    let trs_str = trs.to_string().lines().join(" ");
    format!(
        "\"{}\",{},{},{},{},{},\"{}\"",
        problem, run, order, trial, time, count, trs_str
    )
}

fn update_data<'a, 'b, R: Rng>(
    manager: &mut MCTSManager<TRSMCTS<'a, 'b>>,
    data: &'b [&'b TRSDatum],
    rng: &mut R,
) {
    // 1. Update the known data.
    // notice("updating data", 1);
    manager.tree_mut().mcts_mut().data = data;

    // 2. Update the tree structure.
    // notice("updating tree", 1);
    manager.tree_mut().check_tree(rng);

    // 3. Update the hypotheses: recompute the posterior.
    // notice("updating hypotheses", 1);
    manager.tree_mut().mcts_mut().update_hypotheses();

    // 4. Update the state evaluations.
    //    - Copy the evaluation information to nodes from objects.
    //    - Iteratively update the Q values (the root Q is the sum over the child Qs).
    // notice("updating state evaluations", 1);
    manager.tree_mut().reevaluate_states();
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
    let move_eval = ThompsonMoveEvaluator;
    let root = mcts.root();
    MCTSManager::new(mcts, root, state_eval, move_eval, rng)
}
