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
use generational_arena::{Arena, Index};
use itertools::Itertools;
use list_routine_learning_rs::*;
use polytype::atype::with_ctx;
use programinduction::{
    trs::{
        mcts::{MCTSModel, MCTSObj, MCTSStateEvaluator, MaxThompsonMoveEvaluator, TRSMCTS},
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
            format!("problem,run,order,trial,steps,hypotheses,tree,dag,search_time,total_time"),
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

fn process_prediction(
    data: &[TRSDatum],
    query: &Rule,
    hyps: &Arena<Hyp>,
    params: &Params,
    predictions: &mut Predictions,
) -> bool {
    let n_hyps = hyps.len();
    let input = &query.lhs;
    let output = &query.rhs[0];
    let trs = best_so_far(hyps.iter(), data);
    let prediction = make_prediction(trs, input, params);
    let correct = prediction == *output;
    predictions.push((correct as usize, n_hyps, trs.to_string()));
    correct
}

fn best_so_far_pair<'a, 'b, 'c, I>(hyps: I, data: &[TRSDatum]) -> (Index, &'c Hyp<'a, 'b>)
where
    I: Iterator<Item = (Index, &'c Hyp<'a, 'b>)>,
{
    let (general, specific): (Vec<_>, Vec<_>) = hyps
        .map(|(x, y)| (x, y))
        .partition(|(_, hyp)| hyp.model.generalizes(data));
    if !general.is_empty() {
        general
            .into_iter()
            .max_by(|(_, x), (_, y)| {
                x.lposterior
                    .partial_cmp(&y.lposterior)
                    .or_else(|| Some(Ordering::Less))
                    .unwrap()
            })
            .unwrap()
    } else {
        specific
            .into_iter()
            .max_by(|(_, x), (_, y)| {
                x.lposterior
                    .partial_cmp(&y.lposterior)
                    .or_else(|| Some(Ordering::Less))
                    .unwrap()
            })
            .unwrap()
    }
}

fn best_so_far<'a, 'b, 'c, I>(pop: I, data: &[TRSDatum]) -> &'c TRS<'a, 'b>
where
    I: Iterator<Item = (Index, &'c Hyp<'a, 'b>)>,
{
    &best_so_far_pair(pop, data).1.object.trs
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
    let mut n_hyps = 0;
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
        update_data(
            &mut manager,
            &trs_data[n_data],
            params.simulation.top_n,
            rng,
        );
        let now = Instant::now();
        manager.tree_mut().mcts_mut().start_trial();
        n_step = manager.step_until(rng, |_| now.elapsed().as_secs_f64() > (timeout as f64));
        manager.tree_mut().mcts_mut().finish_trial();
        record_hypotheses(
            &manager.tree().mcts().hypotheses,
            n_hyps,
            best_fd,
            prediction_fd,
            reservoir,
            problem,
            run,
            order,
            n_data + 1,
            &trs_data_owned[n_data],
            rng,
        )?;
        n_hyps = manager.tree().mcts().hypotheses.len();
        // Make a prediction.
        let hyps = &manager.tree().mcts().hypotheses;
        let current_data = &trs_data_owned[n_data];
        let query = &data[n_data];
        let correct = process_prediction(current_data, query, hyps, params, predictions);
        update_timeout(
            correct,
            &mut timeout,
            params.simulation.timeout,
            params.simulation.confidence,
        );
        notice(
            format!(
                "{},{},{},{},{},{},{},{},{},{}",
                problem,
                run,
                order,
                n_data + 1,
                n_step,
                n_hyps,
                manager.tree().tree().tree_size(),
                manager.tree().tree().dag_size(),
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
    hypotheses: &Arena<Hyp>,
    n: usize,
    best_fd: &mut std::fs::File,
    prediction_fd: &mut std::fs::File,
    reservoir: &mut Reservoir<String>,
    problem: &str,
    run: usize,
    order: usize,
    trial: usize,
    data: &[TRSDatum],
    rng: &mut R,
) -> Result<(), String> {
    // best
    let i = best_so_far_pair(
        hypotheses
            .iter()
            .sorted_by_key(|(x, _)| {
                let (x1, x2) = x.into_raw_parts();
                (x2, x1)
            })
            .take(n.max(1)),
        data,
    )
    .0;
    let best = hypotheses
        .iter()
        .sorted_by_key(|(x, _)| {
            let (x1, x2) = x.into_raw_parts();
            (x2, x1)
        })
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
            hypotheses[i].object.count,
        ))?;
    }
    // predictions
    let i = best_so_far_pair(hypotheses.iter(), data).0;
    str_err(record_hypothesis(
        prediction_fd,
        problem,
        run,
        order,
        trial,
        &hypotheses[i].object.trs,
        hypotheses[i].object.time,
        hypotheses[i].object.count,
    ))?;
    // all
    for (_, h) in hypotheses
        .iter()
        .sorted_by_key(|(x, _)| {
            let (x1, x2) = x.into_raw_parts();
            (x2, x1)
        })
        .skip(n)
    {
        let h_str = hypothesis_string(
            problem,
            run,
            order,
            trial,
            &h.object.trs,
            h.object.time,
            h.object.count,
        );
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
    top_n: usize,
    rng: &mut R,
) {
    // TODO: this doesn't feel semantically very clean. Refactor.
    // 1. Reset the MCTS store.
    manager.tree_mut().mcts_mut().clear();
    manager.tree_mut().mcts_mut().data = data;
    let root_state = manager.tree_mut().mcts_mut().root();

    // 2. Clear the tree store.
    manager.tree_mut().prune_except_top(top_n, root_state, rng);
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
