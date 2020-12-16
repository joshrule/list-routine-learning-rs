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
        mcts::{BestInSubtreeMoveEvaluator, MCTSStateEvaluator, Move, TRSMCTS},
        Datum as TRSDatum, Lexicon, TRS,
    },
    MCTSManager,
};
use rand::{thread_rng, Rng};
use regex::Regex;
use std::{f64, fs::File, io::BufReader, path::PathBuf, process::exit, str, time::Instant};
use term_rewriting::{Operator, Rule};

fn main() {
    with_ctx(4096, |ctx| {
        let start = Instant::now();
        let rng = &mut thread_rng();
        let (
            params,
            _runs,
            problem_filename,
            _best_filename,
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
        let search_time = exit_err(
            search(
                lex.clone(),
                &background,
                &data[..params.simulation.n_predictions],
                &mut params.clone(),
                (&problem, order),
                rng,
            ),
            "search failed",
        );
        let elapsed = start.elapsed().as_secs_f64();
        report_time(search_time, elapsed);
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

fn search<'ctx, 'b, R: Rng>(
    lex: Lexicon<'ctx, 'b>,
    background: &'b [Rule],
    examples: &'b [Rule],
    params: &mut Params,
    (problem, order): (&str, usize),
    rng: &mut R,
) -> Result<f64, String> {
    let mut top_n = TopN::new(params.simulation.top_n);
    let mut manager = make_manager(lex, background, params, &[], rng);
    let timeout = params.simulation.timeout;
    let data = convert_examples_to_data(examples);
    let borrowed_data = data.iter().collect_vec();
    let now = Instant::now();
    let n_prune = params.simulation.top_n;
    manager.tree_mut().mcts_mut().start_trial();
    update_data(&mut manager, &mut top_n, &borrowed_data, n_prune, rng);
    let n_steps = manager.step_until(rng, |_| now.elapsed().as_secs_f64() > (timeout as f64));
    manager.tree_mut().mcts_mut().finish_trial();
    let n_hyps = manager.tree().mcts().hypotheses.len();
    for (_, hyp) in manager.tree().mcts().hypotheses.iter() {
        if let Some(obj) = SimObj::try_new(hyp, manager.tree().mcts()) {
            print_hypothesis(problem, order, &obj);
            top_n.add(ScoredItem {
                score: -obj.hyp.ln_predict_posterior,
                data: Box::new(obj),
            })
        } else {
            println!("# FAILED: {}", hyp.count);
        }
    }
    println!("# top hypotheses:");
    top_n
        .iter()
        .sorted()
        .enumerate()
        .for_each(|(i, h)| println!("# {},{}", i, hypothesis_string(problem, order, &h.data)));
    println!("#");
    println!("# problem: {}", problem);
    println!("# order: {}", order);
    println!("# steps: {}", n_steps);
    println!("# nodes: {}", manager.tree().tree().tree_size());
    println!("# hypotheses: {}", n_hyps);
    Ok(manager.tree().mcts().search_time)
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
                TRSDatum::Partial(e.rhs().unwrap())
            }
        })
        .collect_vec()
}

fn print_hypothesis(problem: &str, order: usize, h: &SimObj) {
    println!("{}", hypothesis_string(problem, order, h));
}

fn hypothesis_string(problem: &str, order: usize, h: &SimObj) -> String {
    hypothesis_string_inner(
        problem,
        order,
        &h.trs,
        &h.hyp.moves,
        h.hyp.time,
        h.hyp.count,
        &[
            h.hyp.obj_meta,
            h.hyp.obj_trs,
            h.hyp.obj_gen,
            h.hyp.obj_acc,
            h.hyp.ln_predict_posterior,
            h.hyp.ln_search_posterior,
        ],
        None,
    )
}

fn hypothesis_string_inner(
    problem: &str,
    order: usize,
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
            "\"{}\",{},{:.9},{},{},\"{}\",\"{}\"",
            problem, order, time, count, objective_string, trs_str, meta_string,
        ),
        Some(result) => format!(
            "\"{}\",{},{:.9},{},{},{},\"{}\",\"{}\"",
            problem, order, time, count, objective_string, result, trs_str, meta_string,
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
    let move_eval = BestInSubtreeMoveEvaluator;
    let root = mcts.root();
    MCTSManager::new(mcts, root, state_eval, move_eval, rng)
}
