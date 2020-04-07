//! A simple tool for figuring out search space sizing.

use docopt::Docopt;
use itertools::Itertools;
use list_routine_learning_rs::*;
use polytype::TypeSchema;
use programinduction::trs::{
    mcts::{take_mcts_step, TRSMCTS},
    Environment, GenerationLimit, Lexicon, TRS,
};
use rand::{
    seq::{IteratorRandom, SliceRandom},
    thread_rng, Rng,
};
use serde_derive::Deserialize;
use std::collections::HashMap;
use std::{
    f64,
    fs::File,
    io::{BufRead, BufReader},
    path::PathBuf,
    process::exit,
    str,
};
use term_rewriting::{Operator, Rule};

fn main() {
    let rng = &mut thread_rng();
    let (params, problem_dir_dir) = exit_err(load_args(), "parameters");
    println!("problem,move,name,n_data,n_rules,stp,gen,com,rec,var,reg,sam");
    // Sample a problem (# data = 10, since that will eventually be true).
    let problem_dir = select_problem(&problem_dir_dir, rng);
    // Initialize everything.
    let mut lex = exit_err(load_lexicon(&problem_dir), "lexicon");
    let background = exit_err(load_rules(&problem_dir, &mut lex), "background");
    let c = exit_err(identify_concept(&lex), "target concept");
    let mut examples = exit_err(load_data(&problem_dir), "data");
    examples.shuffle(rng);
    let data: Vec<_> = examples
        .iter()
        .take(params.simulation.n_predictions - 1)
        .map(|e| e.to_rule(&lex, c))
        .collect::<Result<Vec<_>, _>>()
        .unwrap_or_else(|_| {
            eprintln!("Data conversion failed.");
            exit(1);
        });
    let sam_rules = sample_all_rules(&lex, params.mcts.max_size);
    let mut trs = TRS::new_unchecked(&lex, true, &background, Vec::new());
    let mcts = TRSMCTS::new(
        lex.clone(),
        &background,
        params.simulation.deterministic,
        &data,
        None,
        params.model,
        params.mcts,
    );
    // For Moves 1--5:
    for mv in 0..5 {
        let mut steps_remaining = 2;
        // figure out how many possible options there are
        let options = compute_branching_factor(&trs, &data, params.mcts.max_size, sam_rules.len());
        // pick one move and let the result be trs
        while steps_remaining != 1 {
            steps_remaining = 2;
            if let Some(new_trs) =
                take_mcts_step(Some(trs.clone()), &mut steps_remaining, &mcts, rng)
            {
                trs = new_trs;
            }
        }
        // record the results.
        println!(
            "{},{},{}",
            &problem_dir[4..],
            mv,
            options.iter().map(|x| x.to_string()).join(","),
        );
    }
}

fn sample_all_rules(lex: &Lexicon, max_size: usize) -> Vec<Rule> {
    let mut sam_lex = lex.clone();
    let schema = TypeSchema::Monotype(sam_lex.context_mut().new_variable());
    let limit = GenerationLimit::TermSize(max_size);
    let mut ctx = sam_lex.context().clone();
    sam_lex.enumerate_rules(&schema, limit, true, &mut ctx)
}

fn select_problem<R: Rng>(problem_file: &str, rng: &mut R) -> String {
    let file = File::open(PathBuf::from(problem_file)).unwrap();
    let problems = std::io::BufReader::new(file).lines();
    problems.choose(rng).unwrap().unwrap()
}

#[derive(Deserialize)]
pub struct SizerArgs {
    pub arg_args_file: String,
    pub arg_problem_file: String,
}

fn load_args() -> Result<(Params, String), String> {
    let args: SizerArgs = Docopt::new("Usage: sim <args-file> <problem-file>")
        .and_then(|d| d.deserialize())
        .unwrap_or_else(|e| e.exit());
    let toml_string = path_to_string(".", &args.arg_args_file)?;
    let problem_file = args.arg_problem_file.clone();
    str_err(toml::from_str(&toml_string).map(|toml| (toml, problem_file)))
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

fn compute_branching_factor<'a, 'b>(
    query: &TRS<'a, 'b>,
    data: &[Rule],
    max_size: usize,
    sam: usize,
) -> Vec<usize> {
    let trs = query.clone();

    let stp = 1;

    let mem = 2_usize.pow(data.len() as u32) - 1;
    let del = 2_usize.pow(trs.len() as u32) - 1;

    let gen = !trs.is_empty() as usize;
    let com = trs.find_all_compositions().len();
    let rec = trs.find_all_recursions().len();
    let var = trs
        .try_all_variablizations()
        .iter()
        .map(|rs| rs.len())
        .sum::<usize>();

    let mut reg = 0;
    let lex = trs.lexicon();
    for rule in &trs.utrs().rules {
        let mut types = HashMap::new();
        let mut ctx = lex.context().clone();
        let mut env = Environment::from_vars(&rule.variables(), &mut ctx);
        if lex.infer_rule(rule, &mut types, &mut env, &mut ctx).is_ok() {
            for (_, place) in rule.subterms() {
                let env = Environment::from_rule(rule, &types, place[0] == 0);
                let schema = types[&place].generalize(&lex.free_vars(&mut ctx));
                let terms = lex.enumerate_terms(&schema, max_size, &env, &ctx);
                let n_terms = terms.len();
                reg += n_terms;
            }
        }
    }

    vec![mem, del, stp, gen, com, rec, var, reg, sam]
}
