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
    trs::{parse_trs, task_by_rewrite, GPLexicon, Lexicon, TRS},
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
    let rng = &mut thread_rng();
    let mut params = exit_err(load_args(), "Failed to load parameters");
    params.gp.population_size += 1; // +1 for memorized hypothesis
    notice("loaded parameters", 0);

    let mut lex = exit_err(
        load_lexicon(
            &params.simulation.problem_dir,
            params.simulation.deterministic,
        ),
        "Failed to load lexicon",
    );
    notice("loaded lexicon", 0);
    notice(&lex, 1);

    let c = exit_err(identify_concept(&lex), "No target concept");
    let mut examples = exit_err(
        load_data(&params.simulation.problem_dir),
        "Problem loading data",
    );
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
    notice(examples.iter().map(|e| format!("{:?}", e)).join("\n"), 1);

    let h_star = exit_err(
        load_h_star(&params.simulation.problem_dir, &mut lex),
        "cannot load H*",
    );
    notice("loaded h_star", 0);
    notice(h_star, 1);

    let gp_lex = GPLexicon::new(&lex);
    let mut pop = exit_err(
        initialize_population(&gp_lex, &params, rng, &data[0].lhs),
        "couldn't initialize population",
    );
    notice("initialized population", 0);

    notice("evolving", 0);
    let mut predictions = Vec::with_capacity(data.len());
    exit_err(
        evolve(
            &data[..params.simulation.n_examples],
            &mut predictions,
            &mut pop,
            &gp_lex,
            &mut params,
            rng,
        ),
        "evolution failed",
    );
    notice("", 0);
    println!("i_N,n_seen,accuracy,input,correct,predicted");
    for (n, (accuracy, n_seen, input, prediction, output)) in predictions.iter().enumerate() {
        println!(
            "{},{},{},\"{}\",\"{}\",\"{}\"",
            n, n_seen, accuracy, input, output, prediction
        );
    }
    notice("", 0);
    notice(
        format!("total elapsed time: {:.3e}s", start.elapsed().as_secs_f64()),
        0,
    );
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
    lex: &GPLexicon,
    params: &Params,
    rng: &mut R,
    input: &Term,
) -> Result<Vec<(TRS, f64)>, String> {
    let schema = lex
        .lexicon
        .infer_term(input, &mut HashMap::new())
        .drop()
        .unwrap();
    let mut rules = Vec::with_capacity(params.gp.population_size);
    while rules.len() < rules.capacity() {
        let mut pop = lex.genesis(&params.genetic, rng, 1, &schema);
        let unique = !rules
            .iter()
            .any(|(ptrs, _): &(TRS, _)| UntypedTRS::alphas(&pop[0].utrs(), &ptrs.utrs()));
        if unique {
            let trs = pop[0].full_utrs();
            let sig = lex.lexicon.signature();
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
                    let sig = lex.lexicon.signature().deep_copy();
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
    let utrs = best_trs.full_utrs();
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

fn process_prediction(
    data: &[Rule],
    n_data: usize,
    pop: &mut Vec<(TRS, f64)>,
    params: &mut Params,
    ceiling: usize,
    predictions: &mut Vec<(usize, usize, String, String, String)>,
    seen: &[TRS],
    lex: &GPLexicon,
) {
    let input = &data[n_data].lhs;
    let output = &data[n_data].rhs[0];
    let prediction = make_prediction(pop, input, params);
    let correct = prediction == *output;
    if correct {
        params.gp.n_delta =
            ((params.gp.n_delta as f64) * params.simulation.confidence).ceil() as usize;
        notice(format!("decreased n_delta to {}", params.gp.n_delta), 2);
    } else {
        params.gp.n_delta = ((params.gp.n_delta as f64) * params.simulation.confidence.recip())
            .ceil()
            .min(ceiling as f64) as usize;
        notice(format!("increased n_delta to {}", params.gp.n_delta), 2);
    }
    predictions.push((
        correct as usize,
        seen.len(),
        input.pretty(&lex.lexicon.signature()),
        prediction.pretty(&lex.lexicon.signature()),
        output.pretty(&lex.lexicon.signature()),
    ));
}

fn evolve<R: Rng>(
    data: &[Rule],
    predictions: &mut Vec<(usize, usize, String, String, String)>,
    pop: &mut Vec<(TRS, f64)>,
    lex: &GPLexicon,
    params: &mut Params,
    rng: &mut R,
) -> Result<(), String> {
    let ceiling = params.gp.n_delta;
    notice("n_data,generation,rank,nlposterior,trs", 1);
    let max_op = lex
        .lexicon
        .signature()
        .operators()
        .into_iter()
        .map(|o| o.id())
        .max()
        .unwrap();
    'data: for n_data in 0..data.len() {
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
        let memorized = TRS::new_unchecked(&lex.lexicon, examples.clone());
        let task = task_by_rewrite(&examples, params.model, &lex.lexicon, examples.to_vec())
            .map_err(|_| "bad task".to_string())?;
        for i in pop.iter_mut() {
            i.1 = (task.oracle)(&lex.lexicon, &i.0);
        }
        let mem_score = (task.oracle)(&lex.lexicon, &memorized);
        let mem_pair = (memorized, mem_score);
        let mut seen = if n_data == 0 {
            pop.iter().map(|(x, _)| x).cloned().collect_vec()
        } else {
            vec![]
        };

        for gen in 0..params.simulation.generations_per_datum {
            if !pop.iter().any(|(x, _)| TRS::is_alpha(&mem_pair.0, x)) {
                pop.sort_by(|x, y| x.1.partial_cmp(&y.1).expect("found NaN"));
                pop.pop();
                pop.push(mem_pair.clone());
                if !seen.iter().any(|x| TRS::is_alpha(&mem_pair.0, x)) {
                    seen.push(mem_pair.0.clone());
                }
                pop.sort_by(|x, y| x.1.partial_cmp(&y.1).expect("found NaN"));
            }
            if gen == 0 && n_data == 0 {
                for (i, (h, lpost)) in pop.iter().enumerate() {
                    notice_flat(
                        format!("{},{},{},{:.4},{:?}", n_data, gen, i, lpost, h.to_string()),
                        1,
                    );
                }
                process_prediction(data, n_data, pop, params, ceiling, predictions, &seen, lex);
                continue 'data;
            }
            let mut used_symbols = pop
                .iter()
                .flat_map(|p| p.0.utrs().operators())
                .map(|o| o.id())
                .collect_vec();
            used_symbols.push(max_op);
            lex.lexicon.contract(&used_symbols);
            lex.clear();
            lex.evolve(&params.genetic, rng, &params.gp, &task, &mut seen, pop);
            for (i, (h, lpost)) in pop.iter().enumerate() {
                notice_flat(
                    format!("{},{},{},{:.4},{:?}", n_data, gen, i, lpost, h.to_string()),
                    1,
                );
            }
        }
        if !pop.iter().any(|(x, _)| TRS::is_alpha(&mem_pair.0, x)) {
            pop.sort_by(|x, y| x.1.partial_cmp(&y.1).expect("found NaN"));
            pop.pop();
            pop.push(mem_pair.clone());
            pop.sort_by(|x, y| x.1.partial_cmp(&y.1).expect("found NaN"));
        }
        process_prediction(data, n_data, pop, params, ceiling, predictions, &seen, lex);
    }
    Ok(())
}
