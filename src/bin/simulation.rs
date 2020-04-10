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
use programinduction::{
    trs::{
        mcts::{MCTSMoveEvaluator, MCTSStateEvaluator, TRSMCTS},
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

    let elapsed = start.elapsed().as_secs_f64();
    report_results(search_time, elapsed, &predictions);
}

fn report_results(search_time: f64, total_time: f64, predictions: &[Prediction]) {
    notice(format!("search time: {:.3e}s", search_time), 0);
    notice(format!("total time: {:.3e}s", total_time), 0);
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
    let lex = trs.lexicon();
    let sig = lex.signature();
    let mut trace = Trace::new(
        &utrs,
        sig,
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
        notice(format!("timeout remains at ceiling of {}s", timeout), 1);
        return;
    }
    let change = if correct { "de" } else { "in" };
    let factor = if correct { scale } else { scale.recip() };
    *timeout = ((*timeout as f64) * factor).ceil().min(ceiling as f64) as usize;
    notice(format!("timeout {}creased to {}s", change, timeout), 1);
}

fn search<'a, 'b, R: Rng>(
    lex: Lexicon<'b>,
    background: &[Rule],
    data: &[Rule],
    predictions: &mut Predictions,
    params: &mut Params,
    rng: &mut R,
) -> Result<f64, String> {
    notice("making manager", 1);
    let mut manager = make_manager(lex, background, params, &[], Some(&data[0].lhs), rng);
    let mut timeout = params.simulation.timeout;
    let mut search_time = 0.0;
    let mut n_seen = 0;
    notice("n,trial,nlposterior,trs", 1);
    for n_data in 0..data.len() {
        notice(format!("n_data: {}", n_data), 0);
        if n_data > 0 {
            update_data(&mut manager, &data[..n_data], Some(&data[n_data].lhs));
        }
        let now = Instant::now();
        manager.step_until(rng, |_| now.elapsed().as_secs_f64() > (timeout as f64));
        search_time += now.elapsed().as_secs_f64();
        for h in manager.tree().mcts().hypotheses.iter().skip(n_seen) {
            let h_str = h.trs.to_string();
            let lpost = h.lposterior;
            notice_flat(
                format!("{},{},{:.4},{:?}", n_seen, n_data + 1, lpost, h_str),
                1,
            );
            n_seen += 1;
        }
        notice(format!("search_time: {:.4}", search_time), 0);
        notice(
            format!("revisions: {}", manager.tree().mcts().revisions.len()),
            0,
        );
        notice(
            format!("terminals: {}", manager.tree().mcts().terminals.len()),
            0,
        );
        // Make a prediction.
        notice("making prediction", 0);
        let trss = &manager.tree().mcts().hypotheses;
        let correct = process_prediction(&data[..=n_data], trss, params, predictions);
        update_timeout(
            correct,
            &mut timeout,
            params.simulation.timeout,
            params.simulation.confidence,
        );
    }
    manager
        .tree()
        .to_file("tree.json")
        .map_err(|_| "Record failed")?;
    Ok(search_time)
}

fn update_data<'a, 'b>(
    manager: &mut MCTSManager<TRSMCTS<'a, 'b>>,
    data: &'a [Rule],
    input: Option<&'a Term>,
) {
    notice("updating data", 1);
    // 0. Add the new data to the MCTS object.
    manager.tree_mut().mcts_mut().data = data;
    manager.tree_mut().mcts_mut().input = input;

    // 1. For each object in the object list, call change_data on the hypothesis.
    notice("updating terminals", 2);
    for hypothesis in manager.tree_mut().mcts_mut().hypotheses.iter_mut() {
        hypothesis.change_data(data, input);
        notice(
            format!(
                "{:.4}\t\"{}\"",
                hypothesis.lposterior,
                hypothesis.trs.to_string().lines().join(" ")
            ),
            3,
        );
    }
    // 2. Copy the evaluation information to nodes from objects.
    // 3. Iteratively update the Q values (the root Q is the sum over the child Qs).
    notice("updating states", 2);
    manager.tree_mut().reevaluate_states();

    // 4. Add moves as needed.
    notice("updating moves", 2);
    manager.tree_mut().update_moves();
}

fn make_manager<'a, 'b, R: Rng>(
    lex: Lexicon<'b>,
    background: &'a [Rule],
    params: &Params,
    data: &'a [Rule],
    input: Option<&'a Term>,
    rng: &mut R,
) -> MCTSManager<TRSMCTS<'a, 'b>> {
    let mut mcts = TRSMCTS::new(
        lex,
        background,
        params.simulation.deterministic,
        data,
        input,
        params.model,
        params.mcts,
    );
    let state_eval = MCTSStateEvaluator;
    let move_eval = MCTSMoveEvaluator;
    let root = mcts.root();
    MCTSManager::new(mcts, root, state_eval, move_eval, rng)
}
