//! [Rust][1] simulations using input/output examples to learn [typed][2] first-order [term rewriting systems][3] that perform list routines.
//!
//! [1]: https://www.rust-lang.org
//! "The Rust Programming Language"
//! [2]: https://en.wikipedia.org/wiki/Hindley%E2%80%93Milner_type_system
//! "Wikipedia - Hindley-Milner Type System"
//! [3]: https://en.wikipedia.org/wiki/Rewriting#Term_rewriting_systems
//! "Wikipedia - Term Rewriting Systems"
// TODO: Run *either* GP or MCTS.
// TODO: update data should invalidate anything with infinitely bad likelihood...

use docopt::Docopt;
use itertools::Itertools;
use list_routine_learning_rs::*;
use programinduction::{
    trs::{
        mcts::{MCTSMoveEvaluator, MCTSState, MCTSStateEvaluator, TRSMCTS},
        Hypothesis, Lexicon, TRS,
    },
    MCTSManager,
};
use rand::{seq::SliceRandom, thread_rng, Rng};
use std::{
    cmp::Ordering, f64, fs::File, io::BufReader, path::PathBuf, process::exit, str, time::Instant,
};
use term_rewriting::{trace::Trace, Operator, Rule, Term};

type Prediction = (usize, usize, String);
type Predictions = Vec<Prediction>;

fn main() {
    let start = Instant::now();
    let rng = &mut thread_rng();
    let (mut params, problem_dir) = exit_err(load_args(), "Failed to load parameters");
    notice("loaded parameters", 0);

    let mut lex = exit_err(load_lexicon(&problem_dir), "Failed to load lexicon");
    notice("loaded lexicon", 0);
    notice(&lex, 1);

    let background = exit_err(
        load_rules(&problem_dir, &mut lex),
        "Failed to load background",
    );
    notice("loaded background", 0);

    let c = exit_err(identify_concept(&lex), "No target concept");
    let mut examples = exit_err(load_data(&problem_dir), "Problem loading data");
    examples.shuffle(rng);
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
    let mut predictions = Vec::with_capacity(data.len());
    let search_time = exit_err(
        search(
            lex,
            &background,
            &data[..params.simulation.n_predictions],
            &mut predictions,
            &mut params,
            rng,
        ),
        "search failed",
    );

    notice(format!("search time: {:.3e}s", search_time), 0);
    let elapsed = start.elapsed().as_secs_f64();
    notice(format!("total time: {:.3e}s", elapsed), 0);
    println!("trial,accuracy,n_seen,program");
    for (n, (accuracy, n_seen, program)) in predictions.iter().enumerate() {
        println!(
            "{},{},{},\"{}\"",
            n + 1,
            accuracy,
            n_seen,
            program.lines().join(" ")
        );
    }
}

fn load_args() -> Result<(Params, String), String> {
    let args: Args = Docopt::new("Usage: sim <args-file> <problem-dir>")
        .and_then(|d| d.deserialize())
        .unwrap_or_else(|e| e.exit());
    let toml_string = path_to_string(".", &args.arg_args_file)?;
    str_err(toml::from_str(&toml_string).map(|toml| (toml, args.arg_problem_dir.clone())))
}

fn load_data(problem_dir: &str) -> Result<Vec<Datum>, String> {
    let path: PathBuf = [problem_dir, "stimuli.json"].iter().collect();
    let file = str_err(File::open(path))?;
    let reader = BufReader::new(file);
    let data: Vec<Datum> = str_err(serde_json::from_reader(reader))?;
    Ok(data)
}

fn identify_concept(lex: &Lexicon) -> Result<Operator, String> {
    str_err(
        lex.has_op(Some("C"), 0)
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
    data: &[Rule],
    seen: &[Hypothesis],
    params: &Params,
    predictions: &mut Predictions,
) -> bool {
    let n_data = data.len() - 1;
    let n_seen = seen.len();
    let input = &data[n_data].lhs;
    let output = &data[n_data].rhs[0];
    let trs = best_so_far(seen);
    let prediction = make_prediction(trs, input, params);
    let correct = prediction == *output;
    predictions.push((correct as usize, n_seen, trs.to_string()));
    correct
}

fn best_so_far<'a, 'b, 'c>(pop: &'c [Hypothesis<'a, 'b>]) -> &'c TRS<'a, 'b> {
    &pop.iter()
        .max_by(|x, y| {
            x.lposterior
                .partial_cmp(&y.lposterior)
                .or_else(|| Some(Ordering::Less))
                .unwrap()
        })
        .unwrap()
        .trs
}

fn make_prediction<'a, 'b>(trs: &TRS<'a, 'b>, input: &Term, params: &Params) -> Term {
    let utrs = trs.full_utrs();
    let sig = trs.lexicon().signature();
    let mut trace = Trace::new(
        &utrs,
        &sig,
        input,
        params.model.likelihood.p_observe,
        params.model.likelihood.max_size,
        params.model.likelihood.max_depth,
        params.model.likelihood.strategy,
    );
    trace.rewrite(params.model.likelihood.max_steps);
    trace
        .root()
        .iter()
        .max_by(|n1, n2| {
            n1.log_p()
                .partial_cmp(&n2.log_p())
                .or(Some(Ordering::Less))
                .unwrap()
        })
        .unwrap()
        .term()
}

fn update_timeout(correct: bool, timeout: &mut usize, ceiling: usize, scale: f64) {
    if !correct && *timeout == ceiling {
        notice(format!("timeout remains at ceiling of {}s", timeout), 2);
        return;
    }
    let change = if correct { "de" } else { "in" };
    let factor = if correct { scale } else { scale.recip() };
    *timeout = ((*timeout as f64) * factor).ceil().min(ceiling as f64) as usize;
    notice(format!("timeout {}creased to {}s", change, timeout), 2);
}

fn search<'a, 'b, R: Rng>(
    lex: Lexicon<'b>,
    background: &[Rule],
    data: &[Rule],
    predictions: &mut Predictions,
    params: &mut Params,
    rng: &mut R,
) -> Result<f64, String> {
    let mut manager = make_manager(lex, background, &[], params);
    let mut timeout = params.simulation.timeout;
    let mut search_time = 0.0;
    let mut n_seen = 0;
    notice("trial,nlposterior,trs", 1);
    for n_data in 0..data.len() {
        notice(format!("n_data: {}", n_data), 0);
        notice("updating data", 0);
        update_data(&mut manager, &data[..n_data]);
        manager.tree().show();
        // TODO: Do we want to do something different on the first trial?
        // manager.step_until(rng, |_| now.elapsed().as_secs_f64() > (timeout as f64));
        for _ in 0..10 {
            let now = Instant::now();
            manager.step(rng);
            search_time += now.elapsed().as_secs_f64();
            for h in manager.tree().mcts().trss.iter().skip(n_seen) {
                let h_str = h.trs.to_string();
                let lpost = h.lposterior;
                notice_flat(format!("{},{:.4},{:?}", n_data + 1, lpost, h_str), 1);
                n_seen += 1;
            }
        }
        println!("search_time: {:.4}", search_time);
        println!("searched: {}", manager.tree().mcts().trss.len());
        // Make a prediction.
        notice("making prediction", 0);
        let correct = process_prediction(
            &data[..n_data + 1],
            &manager.tree().mcts().trss,
            params,
            predictions,
        );
        update_timeout(
            correct,
            &mut timeout,
            params.simulation.timeout,
            params.simulation.confidence,
        );
    }
    Ok(search_time)
}

fn update_data<'a, 'b>(manager: &mut MCTSManager<TRSMCTS<'a, 'b>>, data: &'a [Rule]) {
    // 0. Add the new data to the MCTS object.
    manager.tree_mut().mcts_mut().obs = data;

    // 1. For each object in the object list, call change_data on the hypothesis.
    for h in manager.tree_mut().mcts_mut().trss.iter_mut() {
        h.change_data(data);
    }
    notice("rescored objects", 1);
    // 2. Copy the evaluation information to nodes from objects.
    // 3. Iteratively update the Q values (the root Q is the sum over the child Qs).
    // 4. Copy the node Q to the moves while iterating over the list of moves.
    manager.tree_mut().reevaluate_states();
    notice("reevaluated states", 1);

    // 5. Add moves as needed.
    manager.tree_mut().update_moves();
    notice("updated moves", 1);
}

fn make_manager<'a, 'b>(
    lex: Lexicon<'b>,
    background: &'a [Rule],
    data: &'a [Rule],
    params: &Params,
) -> MCTSManager<TRSMCTS<'a, 'b>> {
    let mcts = TRSMCTS::new(
        lex,
        background,
        data,
        params.model,
        params.mcts.max_depth,
        params.mcts.max_states,
    );
    let state_eval = MCTSStateEvaluator;
    let move_eval = MCTSMoveEvaluator;
    let root = MCTSState::new(vec![], params.mcts.moves.clone());
    MCTSManager::new(mcts, root, state_eval, move_eval)
}
