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
    hypotheses::Bayesable,
    inference::{Control, MCMCChain},
    mcts::MCTSManager,
    trs::{
        mcts::{BestInSubtreeMoveEvaluator, MCTSStateEvaluator, TRSMCTS},
        metaprogram::{MetaProgram, MetaProgramControl, MetaProgramHypothesis, Move, State},
        Datum as TRSDatum, Lexicon, TRS,
    },
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
            out_filename,
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
        //let search_time = exit_err(
        //    search(
        //        lex.clone(),
        //        &background,
        //        &data[..params.simulation.n_predictions],
        //        &mut params.clone(),
        //        (&problem, order),
        //        &out_filename,
        //        rng,
        //    ),
        //    "search failed",
        //);
        let search_time = exit_err(
            search_mcmc(
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
    out_filename: &str,
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
    update_data(
        manager.tree_mut().mcts_mut(),
        &mut top_n,
        &borrowed_data,
        n_prune,
    );
    prune_tree(&mut manager, &mut top_n, rng);
    let n_steps = manager.step_until(rng, |_| now.elapsed().as_secs_f64() > (timeout as f64));
    manager.tree_mut().mcts_mut().finish_trial();
    let n_hyps = manager.tree().mcts().hypotheses.len();
    println!("# END OF SEARCH");
    for (_, hyp) in manager.tree().mcts().hypotheses.iter() {
        if let Some(obj) = SimObj::try_new(hyp, manager.tree().mcts()) {
            print_hypothesis(problem, order, &obj);
            top_n.add(ScoredItem {
                score: -obj.hyp.ln_posterior,
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
    manager.tree().to_file(out_filename).expect("wrote tree");
    Ok(manager.tree().mcts().search_time)
}

#[derive(Clone, PartialEq, Eq)]
pub struct MetaProgramHypothesisWrapper<'ctx, 'b>(MetaProgramHypothesis<'ctx, 'b>);

impl<'ctx, 'b> Keyed for MetaProgramHypothesisWrapper<'ctx, 'b> {
    type Key = MetaProgram<'ctx, 'b>;
    fn key(&self) -> &Self::Key {
        &self.0.state.path
    }
}

fn search_mcmc<'ctx, 'b, R: Rng>(
    lex: Lexicon<'ctx, 'b>,
    background: &'b [Rule],
    examples: &'b [Rule],
    params: &mut Params,
    (problem, order): (&str, usize),
    rng: &mut R,
) -> Result<f64, String> {
    println!("# problem,order,time,count,ln_meta,ln_trs,ln_wf,ln_acc,ln_posterior,trs,meta");
    let now = Instant::now();
    let mut top_n: TopN<Box<MetaProgramHypothesisWrapper>> = TopN::new(params.simulation.top_n);
    // TODO: hacked in constants.
    let mut mpctl = MetaProgramControl::new(&[], &params.model, 7, 50);
    let timeout = params.simulation.timeout;
    let data = convert_examples_to_data(examples);
    let borrowed_data = data.iter().collect_vec();
    // TODO: should go after trial start, but am here to avoid mutability issues.
    update_data_mcmc(
        &mut mpctl,
        &mut top_n,
        &borrowed_data,
        params.simulation.top_n,
    );
    let t0 = TRS::new_unchecked(&lex, params.simulation.deterministic, background, vec![]);
    let p0 = MetaProgram::from(t0);
    let h0 = MetaProgramHypothesis::new(&mpctl, p0);
    let mut ctl = Control::new(0, timeout * 1000, 0, 0, 0);
    // TODO: fix me
    //mcts.start_trial();
    let mut chain = MCMCChain::new(h0, &borrowed_data);
    {
        //let mut chain_iter = chain.iter(ctl, rng);
        println!("# drawing samples: {}ms", now.elapsed().as_millis());
        while let Some(sample) = chain.internal_next(&mut ctl, rng) {
            print_hypothesis_mcmc(problem, order, &sample);
            top_n.add(ScoredItem {
                score: -sample.bayes_score().posterior,
                data: Box::new(MetaProgramHypothesisWrapper(sample.clone())),
            })
        }
    }
    // TODO: fix me
    // mcts.finish_trial();
    println!("# END OF SEARCH");
    println!("# top hypotheses:");
    top_n.iter().sorted().enumerate().for_each(|(i, h)| {
        println!(
            "# {},{}",
            i,
            hypothesis_string_mcmc(problem, order, &h.data.0)
        )
    });
    println!("#");
    println!("# problem: {}", problem);
    println!("# order: {}", order);
    println!("# hypotheses: {}", chain.samples());
    println!("# ratio: {}", chain.acceptance_ratio());
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
                TRSDatum::Partial(e.rhs().unwrap())
            }
        })
        .collect_vec()
}

fn print_hypothesis(problem: &str, order: usize, h: &SimObj) {
    println!("{}", hypothesis_string(problem, order, h));
}

fn print_hypothesis_mcmc(problem: &str, order: usize, h: &MetaProgramHypothesis) {
    println!("{}", hypothesis_string_mcmc(problem, order, h));
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
            h.hyp.ln_meta,
            h.hyp.ln_trs,
            h.hyp.ln_wf,
            h.hyp.ln_acc,
            h.hyp.ln_posterior,
            h.hyp.ln_posterior,
        ],
        None,
    )
}

fn hypothesis_string_mcmc(problem: &str, order: usize, h: &MetaProgramHypothesis) -> String {
    hypothesis_string_inner(
        problem,
        order,
        h.state.trs().unwrap(),
        &h.state.metaprogram().unwrap().iter().cloned().collect_vec(),
        h.birth.time,
        h.birth.count,
        &[
            h.ln_meta,
            h.ln_trs,
            h.ln_wf,
            h.ln_acc,
            h.bayes_score().posterior,
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
            "\"{}\",{},{},{},{},\"{}\",\"{}\"",
            problem, order, time, count, objective_string, trs_str, meta_string,
        ),
        Some(result) => format!(
            "\"{}\",{},{},{},{},{},\"{}\",\"{}\"",
            problem, order, time, count, objective_string, result, trs_str, meta_string,
        ),
    }
}

fn prune_tree<'a, 'b, R: Rng>(
    manager: &mut MCTSManager<TRSMCTS<'a, 'b>>,
    top_n: &mut TopN<Box<SimObj<'a, 'b>>>,
    rng: &mut R,
) {
    // Prune the tree store.
    let root_state = manager.tree_mut().mcts_mut().root();
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

fn update_data<'a, 'b>(
    mcts: &mut TRSMCTS<'a, 'b>,
    top_n: &mut TopN<Box<SimObj<'a, 'b>>>,
    data: &'b [&'b TRSDatum],
    prune_n: usize,
) {
    // 0. Update the data.
    mcts.ctl.data = data;

    // 1. Update the top_n.
    for mut h in std::mem::replace(top_n, TopN::new(prune_n)).to_vec() {
        h.data.update_posterior(&mcts.ctl);
        h.score = -h.data.hyp.ln_posterior;
        top_n.add(h);
    }

    // 2. Reset the MCTS store.
    mcts.clear();
}

fn update_data_mcmc<'a, 'b>(
    ctl: &mut MetaProgramControl<'b>,
    top_n: &mut TopN<Box<MetaProgramHypothesisWrapper<'a, 'b>>>,
    data: &'b [&'b TRSDatum],
    prune_n: usize,
) {
    // 0. Update the data.
    ctl.data = data;

    // 1. Update the top_n.
    for mut h in std::mem::replace(top_n, TopN::new(prune_n)).to_vec() {
        h.data.0.compute_posterior(ctl.data, None);
        h.score = -h.data.0.bayes_score().posterior;
        top_n.add(h);
    }
}

fn make_mcts<'ctx, 'b>(
    lex: Lexicon<'ctx, 'b>,
    background: &'b [Rule],
    params: &'b Params,
    data: &'b [&'b TRSDatum],
) -> TRSMCTS<'ctx, 'b> {
    // TODO: hacked in constants
    let mpctl = MetaProgramControl::new(data, &params.model, 7, 50);
    TRSMCTS::new(
        lex,
        background,
        params.simulation.deterministic,
        params.simulation.lo,
        params.simulation.hi,
        mpctl,
        params.mcts,
    )
}

fn make_manager<'ctx, 'b, R: Rng>(
    lex: Lexicon<'ctx, 'b>,
    background: &'b [Rule],
    params: &'b Params,
    data: &'b [&'b TRSDatum],
    rng: &mut R,
) -> MCTSManager<TRSMCTS<'ctx, 'b>> {
    let mut mcts = make_mcts(lex, background, params, data);
    let state_eval = MCTSStateEvaluator;
    let move_eval = BestInSubtreeMoveEvaluator;
    let root = mcts.root();
    MCTSManager::new(mcts, root, state_eval, move_eval, rng)
}
