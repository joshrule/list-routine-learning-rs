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
    trs::{task_by_rewrite, GPLexicon, GeneticParamsFull, Lexicon, TRS},
    GP,
};
use rand::{seq::SliceRandom, thread_rng, Rng};
use std::{
    cmp::Ordering, collections::HashMap, f64, fs::File, io::BufReader, path::PathBuf,
    process::exit, str, time::Instant,
};
use term_rewriting::{trace::Trace, Operator, Rule, Term, TRS as UntypedTRS};

fn main() {
    let start = Instant::now();
    let rng = &mut thread_rng();
    let (mut params, problem_dir) = exit_err(load_args(), "Failed to load parameters");
    params.gp.population_size += 1; // +1 for memorized hypothesis
    notice("loaded parameters", 0);

    let mut lex = exit_err(load_lexicon(&problem_dir), "Failed to load lexicon");
    notice("loaded lexicon", 0);
    notice(&lex, 1);

    let templates = exit_err(
        load_templates(&problem_dir, &mut lex),
        "Failed to load templates",
    );
    notice("loaded templates", 0);

    let background = exit_err(
        load_background(&problem_dir, &mut lex),
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
    notice(examples.iter().map(|e| format!("{:?}", e)).join("\n"), 1);

    let mut gp_lex = GPLexicon::new(&lex, &background, templates);
    notice("created GPLexicon", 0);
    let mut pop = exit_err(
        initialize_population(&gp_lex, &params, rng, &data[0].lhs),
        "couldn't initialize population",
    );
    notice("initialized population", 0);

    notice("evolving", 0);
    let mut predictions = Vec::with_capacity(data.len());
    exit_err(
        evolve(
            &data[..params.simulation.n_predictions],
            &mut predictions,
            &mut pop,
            &mut gp_lex,
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

//fn load_h_star<'a, 'b>(
//    problem_dir: &str,
//    lex: &mut Lexicon<'b>,
//    deterministic: bool,
//    bg: &'a [Rule],
//) -> Result<TRS<'a, 'b>, String> {
//    str_err(parse_trs(
//        &path_to_string(problem_dir, "evaluate")?,
//        lex,
//        deterministic,
//        bg,
//    ))
//}

pub fn logsumexp(lps: &[f64]) -> f64 {
    let largest = lps.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let x = lps.iter().map(|lp| (lp - largest).exp()).sum::<f64>().ln();
    largest + x
}

fn initialize_population<'a, 'b, R: Rng>(
    lex: &GPLexicon<'a, 'b>,
    params: &Params,
    rng: &mut R,
    input: &Term,
) -> Result<Vec<(TRS<'a, 'b>, f64)>, String> {
    let schema = lex
        .lexicon
        .infer_term(input, &mut HashMap::new())
        .drop()
        .unwrap();
    // notice("inferred", 1);
    let mut rules = Vec::with_capacity(params.gp.population_size);
    // notice("entering sampling loop", 1);
    let mut count = 0;
    while rules.len() < rules.capacity() {
        count += 1;
        // notice("another round", 2);
        let g_full = GeneticParamsFull::new(&params.genetic, params.model);
        let mut pop = lex.genesis(&g_full, rng, 1, &schema);
        // notice("genesis", 3);
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
                params.model.likelihood.p_observe,
                params.model.likelihood.max_size,
                params.model.likelihood.max_depth,
                params.model.likelihood.strategy,
            );
            trace.rewrite(params.model.likelihood.max_steps);
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
            // notice("mass computed", 3);
            if mass.is_finite() {
                let trs = pop.pop().unwrap();
                let prior = trs.log_prior(params.model.prior);
                rules.push((trs, -prior));
                // notice("pushed", 3);
            }
        }
    }
    // notice("sorting", 1);
    notice(
        format!(
            "initial population of {} required {} samples",
            rules.len(),
            count,
        ),
        1,
    );
    rules.sort_by(|x, y| {
        x.1.partial_cmp(&y.1)
            .or(Some(std::cmp::Ordering::Equal))
            .unwrap()
    });
    Ok(rules)
}

fn make_prediction<'a, 'b, 'c>(
    pop: &'c [(TRS<'a, 'b>, f64)],
    input: &Term,
    params: &Params,
) -> (Term, &'c TRS<'a, 'b>) {
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
        params.model.likelihood.p_observe,
        params.model.likelihood.max_size,
        params.model.likelihood.max_depth,
        params.model.likelihood.strategy,
    );
    trace.rewrite(params.model.likelihood.max_steps);
    let term = trace
        .root()
        .iter()
        .max_by(|n1, n2| {
            n1.log_p()
                .partial_cmp(&n2.log_p())
                .or(Some(Ordering::Less))
                .unwrap()
        })
        .unwrap()
        .term();
    (term, best_trs)
}

#[allow(clippy::too_many_arguments)]
fn process_prediction(
    data: &[Rule],
    n_data: usize,
    pop: &mut Vec<(TRS, f64)>,
    params: &mut Params,
    ceiling: usize,
    predictions: &mut Vec<(usize, usize, String, String, String)>,
    seen: &[TRS],
) {
    let input = &data[n_data].lhs;
    let output = &data[n_data].rhs[0];
    let (prediction, trs) = make_prediction(pop, input, params);
    let correct = prediction == *output;
    update_confidence(correct, ceiling, params);
    predictions.push((
        correct as usize,
        seen.len(),
        input.pretty(&trs.lexicon().signature()),
        prediction.pretty(&trs.lexicon().signature()),
        output.pretty(&trs.lexicon().signature()),
    ));
}

fn update_confidence(correct: bool, ceiling: usize, params: &mut Params) {
    if correct {
        params.simulation.generations_per_datum = ((params.simulation.generations_per_datum as f64)
            * params.simulation.confidence)
            .ceil() as usize;
        notice(
            format!(
                "decreased generations_per_datum to {}",
                params.simulation.generations_per_datum
            ),
            2,
        );
    } else {
        params.simulation.generations_per_datum = ((params.simulation.generations_per_datum as f64)
            / params.simulation.confidence)
            .ceil()
            .min(ceiling as f64) as usize;
        notice(
            format!(
                "increased generations_per_datum to {}",
                params.simulation.generations_per_datum
            ),
            2,
        );
    }
}

fn evolve<'a, 'b, R: Rng>(
    data: &[Rule],
    predictions: &mut Vec<(usize, usize, String, String, String)>,
    pop: &mut Vec<(TRS<'a, 'b>, f64)>,
    lex: &mut GPLexicon<'a, 'b>,
    params: &mut Params,
    rng: &mut R,
) -> Result<(), String> {
    let ceiling = params.simulation.generations_per_datum;
    notice("n_data,generation,rank,nlposterior,trs", 1);
    let max_op = lex
        .lexicon
        .signature()
        .operators()
        .iter()
        .map(|o| o.id())
        .max()
        .unwrap();
    let mut t = 1.0;
    'data: for n_data in 0..data.len() {
        let examples = data[..n_data].to_vec();
        let memorized = TRS::new_unchecked(
            &lex.lexicon,
            params.genetic.deterministic,
            &lex.bg,
            examples.clone(),
        );
        let task = task_by_rewrite(&examples, params.model, &lex.lexicon, t, examples.to_vec())
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
                // Only use memorized hypothesis when it contains rules.
                if n_data > 0 {
                    pop.pop();
                    pop.push(mem_pair.clone());
                    if !seen.iter().any(|x| TRS::is_alpha(&mem_pair.0, x)) {
                        seen.push(mem_pair.0.clone());
                    }
                    pop.sort_by(|x, y| x.1.partial_cmp(&y.1).expect("found NaN"));
                }
            }
            if gen == 0 && n_data == 0 {
                for (i, (h, lpost)) in pop.iter().enumerate() {
                    notice_flat(
                        format!("{},{},{},{:.4},{:?}", n_data, gen, i, lpost, h.to_string()),
                        1,
                    );
                }
                process_prediction(data, n_data, pop, params, ceiling, predictions, &seen);
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
            let g_full = GeneticParamsFull::new(&params.genetic, params.model);
            lex.evolve(&g_full, rng, &params.gp, &task, &mut seen, pop);
            for (i, (h, lpost)) in pop.iter().enumerate() {
                notice_flat(
                    format!("{},{},{},{:.4},{:?}", n_data, gen, i, lpost, h.to_string()),
                    1,
                );
            }
            t += 1.0;
        }
        if !pop.iter().any(|(x, _)| TRS::is_alpha(&mem_pair.0, x)) {
            pop.sort_by(|x, y| x.1.partial_cmp(&y.1).expect("found NaN"));
            pop.pop();
            pop.push(mem_pair.clone());
            pop.sort_by(|x, y| x.1.partial_cmp(&y.1).expect("found NaN"));
        }
        process_prediction(data, n_data, pop, params, ceiling, predictions, &seen);
    }
    Ok(())
}
