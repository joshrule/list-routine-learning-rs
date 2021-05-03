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
    mcts::MCTSManager,
    trs::{
        mcts::{BestInSubtreeMoveEvaluator, MCTSStateEvaluator, TRSMCTS},
        metaprogram::{
            MetaProgram, MetaProgramControl, MetaProgramHypothesis, Move, State, StateLabel,
            Temperature,
        },
        Datum as TRSDatum, Lexicon, TRS,
    },
};
use rand::{thread_rng, Rng};
use regex::Regex;
use std::{
    cmp::Ordering, f64, fs::File, io::BufReader, path::PathBuf, process::exit, str, time::Instant,
};
use term_rewriting::{trace::Trace, Operator, Rule, Term};

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
                &data,
                params.simulation.n_predictions,
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
            print_hypothesis(problem, order, &obj, false);
            top_n.add(ScoredItem {
                score: -obj.hyp.ln_posterior,
                data: Box::new(obj),
            })
        } else {
            println!("# FAILED: {}", hyp.count);
        }
    }
    println!("# top hypotheses:");
    top_n.iter().sorted().enumerate().for_each(|(i, h)| {
        println!(
            "# {}\t{}",
            i,
            hypothesis_string(problem, order, &h.data, true)
        )
    });
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
    mut lex: Lexicon<'ctx, 'b>,
    background: &'b [Rule],
    examples: &'b [Rule],
    train_set_size: usize,
    params: &mut Params,
    (problem, order): (&str, usize),
    rng: &mut R,
) -> Result<f64, String> {
    println!("# problem,order,time,count,ln_meta,ln_trs,ln_wf,ln_acc,ln_posterior,trs,meta");
    let now = Instant::now();
    let mut top_n: TopN<Box<MetaProgramHypothesisWrapper>> = TopN::new(params.simulation.top_n);
    // TODO: hacked in constants.
    let mut mpctl = MetaProgramControl::new(&[], &params.model, params.mcts.atom_weights, 7, 50);
    let timeout = params.simulation.timeout;
    let train_data = convert_examples_to_data(&examples[..train_set_size]);
    let borrowed_data = train_data.iter().collect_vec();
    // TODO: should go after trial start, but am here to avoid mutability issues.
    update_data_mcmc(
        &mut mpctl,
        &mut top_n,
        &borrowed_data,
        params.simulation.top_n,
    );
    let mut t0 = TRS::new_unchecked(&lex, params.simulation.deterministic, background, vec![]);
    t0.set_bounds(params.simulation.lo, params.simulation.hi);
    t0.identify_symbols();
    let p0 = MetaProgram::from(t0);
    let h0 = MetaProgramHypothesis::new(&mpctl, p0);
    let mut ctl = Control::new(0, timeout * 1000, 0, 0, 0);
    // TODO: fix me
    //mcts.start_trial();
    let swap = 5000;
    let ladder = TemperatureLadder(vec![Temperature::new(1.0, 1.0)]);
    let mut chain = ParallelTempering::new(h0, &borrowed_data, ladder, swap, rng);
    {
        //let mut chain_iter = chain.iter(ctl, rng);
        println!("# drawing samples: {}ms", now.elapsed().as_millis());
        while let Some(sample) = chain.internal_next(&mut ctl, rng) {
            print_hypothesis_mcmc(problem, order, &sample, true);
            top_n.add(ScoredItem {
                // TODO: magic constant.
                score: -sample.at_temperature(Temperature::new(2.0 / 3.0, 0.04)),
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
            hypothesis_string_mcmc(problem, order, &h.data.0, true)
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
            TRSDatum::Full(e)
            // if i < examples.len() - 1 {
            //     TRSDatum::Full(e)
            // } else {
            //     TRSDatum::Partial(e.lhs)
            // }
        })
        .collect_vec()
}

fn print_hypothesis(problem: &str, order: usize, h: &SimObj, print_trs: bool) {
    println!("{}", hypothesis_string(problem, order, h, print_trs));
}

fn print_hypothesis_mcmc(problem: &str, order: usize, h: &MetaProgramHypothesis, print_trs: bool) {
    println!("{}", hypothesis_string_mcmc(problem, order, h, print_trs));
}

fn hypothesis_string(problem: &str, order: usize, h: &SimObj, print_trs: bool) -> String {
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
        print_trs,
    )
}

fn hypothesis_string_mcmc(
    problem: &str,
    order: usize,
    h: &MetaProgramHypothesis,
    print_trs: bool,
) -> String {
    hypothesis_string_inner(
        problem,
        order,
        h.state.trs().unwrap(),
        &h.state.metaprogram().unwrap().iter().cloned().collect_vec(),
        h.birth.time,
        h.birth.count,
        &[
            // TODO: fixme
            h.ln_meta,
            h.ln_trs,
            h.ln_wf,
            h.ln_acc,
            h.at_temperature(Temperature::new(2.0 / 3.0, 0.04)),
        ],
        None,
        print_trs,
    )
}

fn hypothesis_string_inner(
    _problem: &str,
    _order: usize,
    trs: &TRS,
    moves: &[Move],
    _time: f64,
    _count: usize,
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
        format!("{}\t{}\t\"{}\"\t", trs_len, objective_string, trs_string)
    } else {
        format!(
            "{}\t{}\t\"{}\"\t\"{}\"",
            trs_len, objective_string, trs_string, meta_string
        )
    }
    // TODO: fixme
    //match correct {
    //    None => format!(
    //        "\"{}\",{},{},{},{},\"{}\",\"{}\"",
    //        problem, order, time, count, objective_string, trs_str, meta_string,
    //    ),
    //    Some(result) => format!(
    //        "\"{}\",{},{},{},{},{},\"{}\",\"{}\"",
    //        problem, order, time, count, objective_string, result, trs_str, meta_string,
    //    ),
    //}
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
        h.score = -h.data.0.at_temperature(Temperature::new(2.0 / 3.0, 0.04));
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
    let mpctl = MetaProgramControl::new(data, &params.model, params.mcts.atom_weights, 7, 50);
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

fn process_prediction<'ctx, 'b>(query: &Rule, best: &TRS<'ctx, 'b>, params: &Params) -> bool {
    query.rhs[0] == make_prediction(best, &query.lhs, params)
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
        params.model.likelihood.representation,
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
