use itertools::Itertools;
use polytype::{ptp, tp, Context as TypeContext, TypeSchema};
use programinduction::{
    trs::{parse_lexicon, GeneticParams, Lexicon, ModelParams},
    GPParams,
};
use serde_derive::{Deserialize, Serialize};
use std::{fs::read_to_string, path::PathBuf, process::exit};
use term_rewriting::{Operator, Rule, Term};

pub fn start_section(s: &str) {
    println!("#\n# {}\n# {}", s, "-".repeat(s.len()));
}

pub fn notice_flat<T: std::fmt::Display>(t: T, n: usize) {
    println!("#{}{}", " ".repeat(1 + 2 * n), t);
}

pub fn notice<T: std::fmt::Display>(t: T, n: usize) {
    println!(
        "{}",
        t.to_string()
            .lines()
            .map(|s| format!("#{}{}", " ".repeat(1 + 2 * n), s))
            .join("\n")
    );
}

pub fn exit_err<T>(x: Result<T, String>, msg: &str) -> T {
    x.unwrap_or_else(|err| {
        eprintln!("# {}: {}", msg, err);
        exit(1);
    })
}

pub fn str_err<T, U: ToString>(x: Result<T, U>) -> Result<T, String> {
    x.or_else(|err| Err(err.to_string()))
}

pub fn path_to_string(dir: &str, file: &str) -> Result<String, String> {
    let path: PathBuf = [dir, file].iter().collect();
    str_err(read_to_string(path))
}

pub fn load_lexicon(problem_dir: &str, deterministic: bool) -> Result<Lexicon, String> {
    str_err(parse_lexicon(
        &path_to_string(problem_dir, "signature")?,
        &path_to_string(problem_dir, "background")?,
        &path_to_string(problem_dir, "templates")?,
        deterministic,
        TypeContext::default(),
    ))
}

#[derive(Deserialize)]
pub struct Args {
    pub arg_args_file: String,
}

#[derive(Deserialize)]
pub struct Params {
    pub simulation: SimulationParams,
    pub genetic: GeneticParams,
    pub gp: GPParams,
    pub model: ModelParams,
}

#[derive(Serialize, Deserialize)]
pub struct SimulationParams {
    pub generations_per_datum: usize,
    pub decay: f64,
    pub problem_dir: String,
    pub deterministic: bool,
    pub n_examples: usize,
    pub confidence: f64,
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
                op: lex.has_op(Some("true"), 0).map_err(|_| ())?,
                args: vec![],
            },
            Value::Bool(false) => Term::Application {
                op: lex.has_op(Some("false"), 0).map_err(|_| ())?,
                args: vec![],
            },
        };
        if let Some(op) = lhs {
            let op_term = Term::Application { op, args: vec![] };
            Ok(Term::Application {
                op: lex.has_op(Some("."), 2).map_err(|_| ())?,
                args: vec![op_term, base_term],
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
        let nil = lex.has_op(Some("NIL"), 0).map_err(|_| ())?;
        let cons = lex.has_op(Some("CONS"), 0).map_err(|_| ())?;
        let app = lex.has_op(Some("."), 2).map_err(|_| ())?;
        let mut term = Term::Application {
            op: nil,
            args: vec![],
        };
        for t in ts {
            let cons_term = Term::Application {
                op: cons,
                args: vec![],
            };
            let inner_term = Term::Application {
                op: app,
                args: vec![cons_term, t],
            };
            term = Term::Application {
                op: app,
                args: vec![inner_term, term],
            };
        }
        Ok(term)
    }
    fn make_digit(lex: &Lexicon, n: usize) -> Result<Term, ()> {
        let digit_const = Term::Application {
            op: lex.has_op(Some("DIGIT"), 0).map_err(|_| ())?,
            args: vec![],
        };
        let num_const = Term::Application {
            op: lex.has_op(Some(&n.to_string()), 0).map_err(|_| ())?,
            args: vec![],
        };
        Ok(Term::Application {
            op: lex.has_op(Some("."), 2).map_err(|_| ())?,
            args: vec![digit_const, num_const],
        })
    }
    fn num_to_term(lex: &Lexicon, num: usize) -> Result<Term, ()> {
        match num {
            0..=9 => Value::make_digit(lex, num),
            _ => {
                let app = lex.has_op(Some("."), 2).map_err(|_| ())?;
                let decc_term = Term::Application {
                    op: lex.has_op(Some("DECC"), 0).map_err(|_| ())?,
                    args: vec![],
                };
                let num_term = Value::num_to_term(lex, num / 10)?;
                let inner_term = Term::Application {
                    op: app,
                    args: vec![decc_term, num_term],
                };
                let digit_term = Term::Application {
                    op: lex
                        .has_op(Some(&(num % 10).to_string()), 0)
                        .map_err(|_| ())?,
                    args: vec![],
                };
                Ok(Term::Application {
                    op: app,
                    args: vec![inner_term, digit_term],
                })
            }
        }
    }
}
