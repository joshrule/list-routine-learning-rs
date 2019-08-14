//! [Rust][1] simulations using input/output examples to learn [typed][2] first-order [term rewriting systems][3] that perform list routines.
//!
//! [1]: https://www.rust-lang.org
//! "The Rust Programming Language"
//! [2]: https://en.wikipedia.org/wiki/Hindley%E2%80%93Milner_type_system
//! "Wikipedia - Hindley-Milner Type System"
//! [3]: https://en.wikipedia.org/wiki/Rewriting#Term_rewriting_systems
//! "Wikipedia - Term Rewriting Systems"
extern crate docopt;
#[macro_use]
extern crate polytype;
extern crate programinduction;
extern crate rand;
extern crate serde;
#[macro_use]
extern crate serde_derive;
extern crate serde_json;
extern crate term_rewriting;
extern crate toml;

use docopt::Docopt;
use polytype::Context as TypeContext;
use programinduction::trs::{
    parse_lexicon, parse_templates, parse_trs, task_by_rewrite, GeneticParams, Lexicon,
    ModelParams, TRS,
};
use programinduction::{GPParams, GPSelection, GP};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use std::fs::read_to_string;
use std::io;
use std::path::PathBuf;
use std::process::Command;
use std::str;
use term_rewriting::{Operator, Rule, RuleContext};
use utils::*;

fn main() -> io::Result<()> {
    // TODO: anything to compress here?
    let sim_args = load_args()?;
    let rng = &mut SmallRng::from_seed([1u8; 16]);
    let mut lex = load_lexicon(&sim_args.problem_dir, sim_args.deterministic)?;
    let params = initialize_params(sim_args, &mut lex)?;
    let data = load_routine(&lex, &params.sim_params)?;
    let mut pop = initialize_population(&lex, &params, rng)?;
    let h_star = load_h_star(&params.sim_params, &mut lex)?;
    evolve(&data, &mut pop, &h_star, &lex, &params, rng)?;
    let llike = -h_star.log_likelihood(&data, params.model_params);
    let lpost = llike - h_star.pseudo_log_prior();
    report_results(llike, lpost, params.model_params, &data, &pop);
    summarize_results(lpost, &pop);
    Ok(())
}

fn start_section(s: &str) {
    println!("\n{}\n{}", s, "-".repeat(s.len()));
}

fn initialize_population<R: Rng>(
    lex: &Lexicon,
    params: &Params,
    rng: &mut R,
) -> io::Result<Vec<(TRS, f64)>> {
    let task = task_by_rewrite(&[], params.model_params, lex, ())
        .or_else(|_| Err(io::Error::new(io::ErrorKind::Other, "bad datum")))?;
    Ok(lex.init(&params.genetic_params, rng, &params.gp_params, &task))
}

fn load_args() -> io::Result<TOMLArgs> {
    let args: utils::Args = Docopt::new("Usage: sim <args-file>")
        .and_then(|d| d.deserialize())
        .unwrap_or_else(|e| e.exit());
    let sim_file = args.arg_args_file;
    toml::from_str(&read_to_string(sim_file)?)
        .or_else(|_| Err(io::Error::new(io::ErrorKind::Other, "bad datum")))
}

fn load_routine(lex: &Lexicon, params: &SimulationParams) -> io::Result<Vec<Rule>> {
    start_section("Loading Routine");
    let output = Command::new("../list-routines/list_routines_cli")
        .arg("generate-examples")
        .arg("--count")
        .arg(&params.n_data.to_string())
        .arg(&params.routine_name)
        .output()
        .unwrap();
    let data_str = str::from_utf8(&output.stdout).unwrap();
    let data: Vec<Datum> = serde_json::from_str(data_str).unwrap();
    let concept = identify_concept(&lex, &params.routine_name)?;
    Ok(load_data(&lex, &data, &concept)?)
}

fn identify_concept(lex: &Lexicon, routine_name: &str) -> io::Result<Operator> {
    start_section("Concept");
    let concept = lex
        .has_op(Some("C"), 1)
        .or_else(|_| Err(io::Error::new(io::ErrorKind::Other, "No C")))?;
    {
        println!(
            "{}/{}: {}",
            concept.display(),
            concept.arity(),
            routine_name,
        );
    }
    Ok(concept)
}

fn load_lexicon(problem_dir: &str, deterministic: bool) -> io::Result<Lexicon> {
    start_section("Loading lexicon");
    let sig_file: PathBuf = [problem_dir, "signature"].iter().collect();
    let sig_string = read_to_string(sig_file)?;
    let bg_file: PathBuf = [problem_dir, "background"].iter().collect();
    let bg_string = read_to_string(bg_file)?;
    let lex = parse_lexicon(
        &sig_string,
        &bg_string,
        deterministic,
        TypeContext::default(),
    ).or_else(|_| Err(io::Error::new(io::ErrorKind::Other, "cannot parse lexicon")))?;
    println!("{}", lex);
    Ok(lex)
}

fn load_data(lex: &Lexicon, examples: &[Datum], concept: &Operator) -> io::Result<Vec<Rule>> {
    start_section("Reading data");
    let data: Vec<Rule> = examples
        .iter()
        .map(|x| x.to_rule(lex, concept.clone()))
        .collect::<Result<Vec<Rule>, _>>()
        .or_else(|_| Err(io::Error::new(io::ErrorKind::Other, "bad datum")))?;
    for datum in &data {
        println!("{}", datum.pretty());
    }
    Ok(data)
}

fn load_templates(problem_dir: &str, lex: &mut Lexicon) -> io::Result<Vec<RuleContext>> {
    start_section("Reading templates");
    let template_file: PathBuf = [problem_dir, "templates"].iter().collect();
    let template_string = read_to_string(template_file)?;
    let templates = parse_templates(&template_string, lex).or_else(|_| {
        Err(io::Error::new(
            io::ErrorKind::Other,
            "cannot parse templates",
        ))
    })?;
    {
        //let sig = s.signature();
        for template in &templates {
            println!("{}", template.pretty());
        }
    }
    Ok(templates)
}

fn initialize_params(args: TOMLArgs, lex: &mut Lexicon) -> io::Result<Params> {
    Ok(Params {
        genetic_params: GeneticParams {
            max_sample_size: args.max_sample_size,
            n_crosses: args.n_crosses,
            p_add: args.p_add,
            p_keep: args.p_keep,
            templates: load_templates(&args.problem_dir, lex)?,
            atom_weights: (
                args.variable_weight,
                args.constant_weight,
                args.function_weight,
            ),
        },
        sim_params: SimulationParams {
            n_data: args.n_data,
            routine_name: args.routine_name,
            generations_per_datum: args.generations_per_datum,
            problem_dir: args.problem_dir,
            deterministic: args.deterministic,
        },
        gp_params: GPParams {
            selection: match args.selection.as_str() {
                "probabilistic" | "Probabilistic" => GPSelection::Probabilistic,
                _ => GPSelection::Deterministic,
            },
            population_size: args.population_size,
            tournament_size: args.tournament_size,
            mutation_prob: args.mutation_prob,
            n_delta: args.n_delta,
        },
        model_params: ModelParams {
            p_partial: args.p_partial,
            p_observe: args.p_observe,
            max_steps: args.max_steps,
            max_size: args.max_size,
        },
    })
}

fn load_h_star(sim_params: &SimulationParams, lex: &mut Lexicon) -> io::Result<TRS> {
    start_section("Loading H*");
    let h_star_file: PathBuf = [&sim_params.problem_dir, "evaluate"].iter().collect();
    let h_star_string = read_to_string(h_star_file)?;
    let h_star = parse_trs(&h_star_string, lex)
        .or_else(|_| Err(io::Error::new(io::ErrorKind::Other, "cannot parse TRS")))?;
    println!("{}", h_star);
    Ok(h_star)
}

#[cfg_attr(feature = "cargo-clippy", allow(too_many_arguments))]
fn evolve<R: Rng>(
    data: &[Rule],
    pop: &mut Vec<(TRS, f64)>,
    h_star: &TRS,
    lex: &Lexicon,
    params: &Params,
    rng: &mut R,
) -> io::Result<()> {
    start_section("Evolving");
    println!("n_data,generation,id,llikelihood,lprior,score,difference,description");
    for n_data in 0..=(data.len()) {
        let task = task_by_rewrite(&data[0..n_data], params.model_params, lex, ())
            .or_else(|_| Err(io::Error::new(io::ErrorKind::Other, "bad datum")))?;
        for i in pop.iter_mut() {
            i.1 = (task.oracle)(lex, &i.0);
        }
        let h_star_lpost = (task.oracle)(lex, h_star);

        for gen in 0..params.sim_params.generations_per_datum {
            lex.evolve(&params.genetic_params, rng, &params.gp_params, &task, pop);
            for (i, (individual, score)) in pop.iter().enumerate() {
                let llike = -individual.log_likelihood(data, params.model_params);
                let lprior = -individual.pseudo_log_prior();
                println!(
                    "{},{},{},{:.4},{:.4},{:.4},{:.4},{:?}",
                    n_data,
                    gen,
                    i,
                    llike,
                    lprior,
                    score,
                    h_star_lpost - score,
                    individual.to_string(),
                );
            }
        }
    }
    Ok(())
}

fn report_results(
    h_star_llike: f64,
    h_star_lpost: f64,
    model_params: ModelParams,
    data: &[Rule],
    pop: &[(TRS, f64)],
) {
    start_section("Results");
    println!("rank,nlprior,nllike,nlpost,correct,better,difference");
    for (i, (individual, _)) in pop.iter().enumerate() {
        let nlprior = -individual.pseudo_log_prior();
        let nllike = -individual.log_likelihood(&data, model_params);
        let nlpost = nlprior + nllike;
        println!(
            "{},{:.4},{:.4},{:.4},{},{},{:.4},{:?}",
            i,
            nlprior,
            nllike,
            nlpost,
            (nllike <= h_star_llike) as usize,
            (nlpost <= h_star_lpost) as usize,
            h_star_lpost - nlpost,
            individual.to_string(),
        );
    }
}

fn summarize_results(h_star_score: f64, pop: &[(TRS, f64)]) {
    start_section("Summary");
    println!("best_score,best_difference,mean_score,mean_difference");
    let mean_score = pop.iter().map(|x| x.1).sum::<f64>() / (pop.len() as f64);
    println!(
        "{:.4},{:.4},{:.4},{:.4}",
        pop[0].1,
        h_star_score - pop[0].1,
        mean_score,
        h_star_score - mean_score,
    );
}

mod utils {
    use polytype::TypeSchema;
    use programinduction::trs::{GeneticParams, Lexicon, ModelParams};
    use programinduction::GPParams;
    use term_rewriting::{Operator, Rule, Term};

    #[derive(Deserialize)]
    pub struct Args {
        pub arg_args_file: String,
    }

    #[derive(Deserialize)]
    pub struct TOMLArgs {
        pub n_data: usize,
        pub routine_name: String,
        pub generations_per_datum: usize,
        pub problem_dir: String,
        pub deterministic: bool,
        pub max_sample_size: usize,
        pub n_crosses: usize,
        pub p_add: f64,
        pub p_keep: f64,
        pub variable_weight: f64,
        pub constant_weight: f64,
        pub function_weight: f64,
        pub selection: String,
        pub population_size: usize,
        pub tournament_size: usize,
        pub mutation_prob: f64,
        pub n_delta: usize,
        pub p_partial: f64,
        pub p_observe: f64,
        pub max_steps: usize,
        pub max_size: Option<usize>,
    }

    pub struct Params {
        pub sim_params: SimulationParams,
        pub genetic_params: GeneticParams,
        pub gp_params: GPParams,
        pub model_params: ModelParams,
    }

    pub struct SimulationParams {
        pub n_data: usize,
        pub routine_name: String,
        pub generations_per_datum: usize,
        pub problem_dir: String,
        pub deterministic: bool,
    }

    #[derive(Clone, Debug, Serialize, Deserialize)]
    pub struct Routine {
        #[serde(rename = "type")]
        pub tp: RoutineType,
        pub examples: Vec<Datum>,
        pub name: String,
    }

    #[derive(Clone, Debug, Serialize, Deserialize)]
    pub struct Datum {
        i: Value,
        o: Value,
    }
    impl Datum {
        /// Convert a `Datum` to a term rewriting [`Rule`].
        ///
        /// [`Rule`]: ../term_rewriting/struct.Rule.html
        pub fn to_rule(&self, lex: &Lexicon, concept: Operator) -> Result<Rule, ()> {
            let lhs = self.i.to_term(lex, Some(concept))?;
            let rhs = self.o.to_term(lex, None)?;
            Rule::new(lhs, vec![rhs]).ok_or(())
        }
    }

    #[derive(Copy, Clone, Debug, Serialize, Deserialize)]
    pub struct RoutineType {
        #[serde(rename = "input")]
        pub i: IOType,
        #[serde(rename = "output")]
        pub o: IOType,
    }

    #[derive(Copy, Clone, Debug, Serialize, Deserialize)]
    pub enum IOType {
        #[serde(rename = "bool")]
        Bool,
        #[serde(rename = "list-of-int")]
        IntList,
        #[serde(rename = "int")]
        Int,
    }
    impl From<IOType> for TypeSchema {
        fn from(t: IOType) -> Self {
            match t {
                IOType::Bool => ptp!(bool),
                IOType::Int => ptp!(int),
                IOType::IntList => ptp!(list(tp!(int))),
            }
        }
    }

    #[derive(Clone, Debug, Serialize, Deserialize)]
    #[serde(untagged)]
    pub enum Value {
        Int(usize),
        IntList(Vec<usize>),
        Bool(bool),
    }
    impl Value {
        fn to_term(&self, lex: &Lexicon, lhs: Option<Operator>) -> Result<Term, ()> {
            let base_term = match self {
                Value::Int(x) => Value::num_to_term(lex, *x)?,
                Value::IntList(xs) => Value::list_to_term(lex, &xs)?,
                Value::Bool(true) => Term::Application {
                    op: lex.has_op(Some("true"), 0)?,
                    args: vec![],
                },
                Value::Bool(false) => Term::Application {
                    op: lex.has_op(Some("false"), 0)?,
                    args: vec![],
                },
            };
            if let Some(op) = lhs {
                Ok(Term::Application {
                    op,
                    args: vec![base_term],
                })
            } else {
                Ok(base_term)
            }
        }
        fn list_to_term(lex: &Lexicon, xs: &[usize]) -> Result<Term, ()> {
            let ts: Vec<Term> = xs
                .iter()
                .map(|&x| Value::num_to_term(lex, x))
                .rev()
                .collect::<Result<Vec<_>, _>>()?;
            let nil = lex.has_op(Some("NIL"), 0)?;
            let cons = lex.has_op(Some("CONS"), 2)?;
            let mut term = Term::Application {
                op: nil,
                args: vec![],
            };
            for t in ts {
                term = Term::Application {
                    op: cons.clone(),
                    args: vec![t, term],
                };
            }
            Ok(term)
        }
        fn make_digit(lex: &Lexicon, n: usize) -> Result<Term, ()> {
            let digit = lex.has_op(Some("DIGIT"), 1)?;
            let arg_digit = lex.has_op(Some(&n.to_string()), 0)?;
            let arg = Term::Application {
                op: arg_digit,
                args: vec![],
            };
            Ok(Term::Application {
                op: digit,
                args: vec![arg],
            })
        }
        fn num_to_term(lex: &Lexicon, num: usize) -> Result<Term, ()> {
            match num {
                0...9 => Value::make_digit(lex, num),
                _ => {
                    let decc = lex.has_op(Some("DECC"), 2)?;
                    let arg1 = Value::num_to_term(lex, num / 10)?;
                    let arg2_digit = lex.has_op(Some(&(num % 10).to_string()), 0)?;
                    let arg2 = Term::Application {
                        op: arg2_digit,
                        args: vec![],
                    };
                    Ok(Term::Application {
                        op: decc,
                        args: vec![arg1, arg2],
                    })
                }
            }
        }
    }
}
