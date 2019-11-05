//! [Rust][1] simulations using input/output examples to learn [typed][2] first-order [term rewriting systems][3] that perform list routines.
//!
//! [1]: https://www.rust-lang.org
//! "The Rust Programming Language"
//! [2]: https://en.wikipedia.org/wiki/Hindley%E2%80%93Milner_type_system
//! "Wikipedia - Hindley-Milner Type System"
//! [3]: https://en.wikipedia.org/wiki/Rewriting#Term_rewriting_systems
//! "Wikipedia - Term Rewriting Systems"

use docopt::Docopt;
use itertools::Itertools;
use list_routine_learning_rs::*;
use programinduction::{
    trs::{parse_trs, task_by_rewrite, Lexicon, TRS},
    GP,
};
use rand::{
    distributions::{Bernoulli, Distribution},
    seq::SliceRandom,
    thread_rng, Rng,
};
use std::{
    cmp::Ordering, collections::HashMap, convert::TryInto, f64, fs::File, io::BufReader,
    path::PathBuf, process::exit, str, time::Instant,
};
use term_rewriting::{trace::Trace, Operator, Rule, Term, TRS as UntypedTRS};

fn main() {
    let start = Instant::now();
    start_section("Loading parameters");
    let rng = &mut thread_rng();
    let params = exit_err(load_args(), "Failed to load parameters");

    start_section("Loading lexicon");
    let mut lex = exit_err(
        load_lexicon(
            &params.simulation.problem_dir,
            params.simulation.deterministic,
        ),
        "Failed to load lexicon",
    );
    println!("{}", lex);
    println!("{:?}", lex.context());

    start_section("Loading data");
    let c = exit_err(identify_concept(&lex), "No target concept");
    let mut examples = exit_err(
        load_data(&params.simulation.problem_dir),
        "Problem loading data",
    );
    examples.shuffle(rng);
    for example in &examples {
        println!("{:?}", example);
    }
    let data: Vec<_> = examples
        .iter()
        .map(|e| e.to_rule(&lex, c))
        .collect::<Result<Vec<_>, _>>()
        .unwrap_or_else(|_| {
            eprintln!("Data conversion failed.");
            exit(1);
        });

    start_section("Loading H*");
    let h_star = exit_err(
        load_h_star(&params.simulation.problem_dir, &mut lex),
        "cannot load H*",
    );
    println!("{}", h_star);

    start_section("Initial Population");
    let mut pop = exit_err(
        initialize_population(&lex, &params, rng, &data[0].lhs),
        "couldn't initialize population",
    );
    for (i, (trs, score)) in pop.iter().enumerate() {
        println!("{}: {:.4} {:?}", i, score, trs.to_string());
    }

    start_section("Initial Prediction");
    let mut predictions = Vec::with_capacity(data.len());
    let prediction = make_prediction(&pop, &data[0].lhs, &params);
    predictions.push((
        (prediction == data[0].rhs().unwrap()) as usize,
        data[0].lhs.pretty(&lex.signature()),
        prediction.pretty(&lex.signature()),
        data[0].rhs[0].pretty(&lex.signature()),
    ));
    // println!(
    //     "*** prediction: {} -> {} ({})",
    //     data[0].lhs.pretty(&lex.signature()),
    //     prediction.pretty(&lex.signature()),
    //     data[0].rhs[0].pretty(&lex.signature())
    // );

    if true {
        start_section("Evolving");
        exit_err(
            evolve(
                &data[..params.simulation.n_examples],
                &mut predictions,
                &mut pop,
                &h_star,
                &lex,
                &params,
                rng,
            ),
            "evolution failed",
        );
    }
    println!();
    println!("n,accuracy,input,output,prediction");
    for (n, (accuracy, input, prediction, output)) in predictions.iter().enumerate() {
        println!(
            "{},{},\"{}\",\"{}\",\"{}\"",
            n, accuracy, input, output, prediction
        );
    }
    println!();
    println!("total elapsed time: {:.3e}s", start.elapsed().as_secs_f64());
}

fn load_args() -> Result<Params, String> {
    let args: Args = Docopt::new("Usage: sim <args-file>")
        .and_then(|d| d.deserialize())
        .unwrap_or_else(|e| e.exit());
    let toml_string = path_to_string(".", &args.arg_args_file)?;
    str_err(toml::from_str(&toml_string))
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

fn load_h_star(problem_dir: &str, lex: &mut Lexicon) -> Result<TRS, String> {
    str_err(parse_trs(&path_to_string(problem_dir, "evaluate")?, lex))
}

pub fn logsumexp(lps: &[f64]) -> f64 {
    let largest = lps.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let x = lps.iter().map(|lp| (lp - largest).exp()).sum::<f64>().ln();
    largest + x
}

fn initialize_population<R: Rng>(
    lex: &Lexicon,
    params: &Params,
    rng: &mut R,
    input: &Term,
) -> Result<Vec<(TRS, f64)>, String> {
    let schema = lex.infer_term(input, &mut HashMap::new()).drop().unwrap();
    let mut rules = Vec::with_capacity(params.gp.population_size);
    while rules.len() < rules.capacity() {
        let mut pop = lex.genesis(&params.genetic, rng, 1, &schema);
        let trs = pop[0].utrs();
        let unique = !rules
            .iter()
            .any(|(ptrs, _): &(TRS, _)| UntypedTRS::alphas(&trs, &ptrs.utrs()));
        if unique {
            let sig = lex.signature();
            let mut trace = Trace::new(
                &trs,
                &sig,
                &input,
                params.model.p_observe,
                params.model.max_size,
                params.model.max_depth,
                params.model.strategy,
            );
            trace.rewrite(params.model.max_steps);
            let masss = trace
                .root()
                .iter()
                .map(|x| {
                    //let mut sig = Signature::default();
                    let sig = lex.signature();
                    if UntypedTRS::convert_list_to_string(&x.term(), &sig).is_some() {
                        x.log_p()
                    } else {
                        f64::NEG_INFINITY
                    }
                })
                .collect_vec();
            let mass = logsumexp(&masss);
            if mass.is_finite() {
                let trs = pop.pop().unwrap();
                let prior = trs.log_prior(params.model);
                rules.push((trs, -prior));
            }
        }
    }
    rules.sort_by(|x, y| {
        x.1.partial_cmp(&y.1)
            .or(Some(std::cmp::Ordering::Equal))
            .unwrap()
    });
    Ok(rules)
}

fn make_prediction(pop: &[(TRS, f64)], input: &Term, params: &Params) -> Term {
    let best_trs = &pop
        .iter()
        .min_by(|(_, x), (_, y)| x.partial_cmp(y).or_else(|| Some(Ordering::Less)).unwrap())
        .unwrap()
        .0;
    let utrs = best_trs.utrs();
    let sig = best_trs.lexicon().signature();
    let mut trace = Trace::new(
        &utrs,
        &sig,
        input,
        params.model.p_observe,
        params.model.max_size,
        params.model.max_depth,
        params.model.strategy,
    );
    trace.rewrite(params.model.max_steps);
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

fn evolve<R: Rng>(
    data: &[Rule],
    predictions: &mut Vec<(usize, String, String, String)>,
    pop: &mut Vec<(TRS, f64)>,
    h_star: &TRS,
    lex: &Lexicon,
    params: &Params,
    rng: &mut R,
) -> Result<(), String> {
    println!("n_data,generation,rank,nlposterior,h_star_nlposterior,trs");
    for n_data in 1..data.len() {
        let examples = {
            let mut exs = vec![];
            for (i, datum) in data.iter().enumerate().take(n_data) {
                let idx = n_data - i - 1;
                let weight = params.simulation.decay.powi(idx.try_into().unwrap());
                let dist = Bernoulli::new(weight).unwrap();
                if dist.sample(rng) {
                    exs.push(datum.clone());
                }
            }
            exs
        };
        let input = &data[n_data].lhs;
        let output = &data[n_data].rhs[0];
        let task = task_by_rewrite(&examples, params.model, lex, examples.to_vec())
            .map_err(|_| "bad task".to_string())?;

        for i in pop.iter_mut() {
            i.1 = (task.oracle)(lex, &i.0);
        }
        let h_star_lpost = (task.oracle)(lex, h_star);

        for gen in 0..params.simulation.generations_per_datum {
            lex.evolve(&params.genetic, rng, &params.gp, &task, pop);
            for (i, (h, lpost)) in pop.iter().enumerate() {
                println!(
                    "{},{},{},{:.4},{:.4},{:?}",
                    n_data,
                    gen,
                    i,
                    lpost,
                    h_star_lpost,
                    h.to_string(),
                );
            }
        }
        let prediction = make_prediction(pop, input, params);
        predictions.push((
            (prediction == *output) as usize,
            input.pretty(&lex.signature()),
            prediction.pretty(&lex.signature()),
            output.pretty(&lex.signature()),
        ));
    }
    Ok(())
}
