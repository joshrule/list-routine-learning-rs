use itertools::Itertools;
use polytype::atype::TypeContext;
use programinduction::trs::{
    mcts::{MCTSObj, MCTSParams, TRSMCTS},
    parse_lexicon, parse_rulecontexts, parse_rules, parse_trs, Lexicon, ModelParams,
    SingleLikelihood, TRS,
};
use rand::Rng;
use serde_derive::{Deserialize, Serialize};
use std::{collections::BinaryHeap, fs::read_to_string, path::PathBuf, process::exit};
use term_rewriting::{Operator, Rule, RuleContext, Term};

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

pub fn load_lexicon<'ctx, 'b>(
    ctx: &TypeContext<'ctx>,
    lex_filename: &str,
) -> Result<Lexicon<'ctx, 'b>, String> {
    str_err(parse_lexicon(
        &str_err(read_to_string(PathBuf::from(lex_filename)))?,
        &ctx,
    ))
}

pub fn load_rulecontexts(problem_dir: &str, lex: &mut Lexicon) -> Result<Vec<RuleContext>, String> {
    str_err(parse_rulecontexts(
        &path_to_string(problem_dir, "templates")?,
        lex,
    ))
}

pub fn load_rules(bg_filename: &str, lex: &mut Lexicon) -> Result<Vec<Rule>, String> {
    str_err(parse_rules(
        &str_err(read_to_string(PathBuf::from(bg_filename)))?,
        lex,
    ))
}

pub fn load_trs<'ctx, 'b>(
    problem_dir: &str,
    lex: &mut Lexicon<'ctx, 'b>,
    deterministic: bool,
    bg: &'b [Rule],
) -> Result<TRS<'ctx, 'b>, String> {
    str_err(parse_trs(
        &path_to_string(problem_dir, "evaluate")?,
        lex,
        deterministic,
        bg,
    ))
}

pub struct ScoredItem<T> {
    pub score: f64,
    pub data: T,
}

pub struct Reservoir<T> {
    data: BinaryHeap<ScoredItem<T>>,
    size: usize,
}

impl<T> ScoredItem<T> {
    pub fn new<R: Rng>(data: T, rng: &mut R) -> Self {
        ScoredItem {
            data,
            score: rng.gen(),
        }
    }
}

impl<T: Eq> PartialEq for ScoredItem<T> {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score && self.data == other.data
    }
}

impl<T: Eq> Eq for ScoredItem<T> {}

impl<T: Eq> PartialOrd for ScoredItem<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(
            self.score
                .partial_cmp(&other.score)
                .unwrap_or(std::cmp::Ordering::Equal),
        )
    }
}

impl<T: Eq> Ord for ScoredItem<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(&other).expect("ordering")
    }
}

impl<T: Eq> Reservoir<T> {
    pub fn with_capacity(n: usize) -> Self {
        Reservoir {
            data: BinaryHeap::with_capacity(n),
            size: n,
        }
    }
    pub fn add<F, R: Rng>(&mut self, f: F, rng: &mut R)
    where
        F: Fn() -> T,
    {
        if self.data.len() < self.size {
            self.data.push(ScoredItem::new(f(), rng));
        } else if let Some(best) = self.data.peek() {
            let item_score = rng.gen();
            if item_score < best.score {
                self.data.pop();
                self.data.push(ScoredItem {
                    score: item_score,
                    data: f(),
                });
            }
        }
    }
    pub fn to_vec(self) -> Vec<ScoredItem<T>> {
        self.data.into_sorted_vec()
    }
}

pub struct TopN<T> {
    pub(crate) size: usize,
    pub(crate) data: BinaryHeap<ScoredItem<T>>,
}

impl<T: Eq> TopN<T> {
    pub fn new(size: usize) -> Self {
        TopN {
            size,
            data: BinaryHeap::with_capacity(size),
        }
    }
    pub fn add(&mut self, datum: ScoredItem<T>) {
        if self.data.len() < self.size {
            self.data.push(datum);
        } else if let Some(best) = self.data.peek() {
            if datum.score > best.score {
                self.data.pop();
                self.data.push(datum);
            }
        }
    }
    pub fn pop(&mut self) -> Option<ScoredItem<T>> {
        self.data.pop()
    }
    pub fn to_vec(self) -> Vec<ScoredItem<T>> {
        self.data.into_sorted_vec()
    }
    pub fn least(&self) -> Option<&ScoredItem<T>> {
        self.data.iter().max()
    }
    pub fn iter<'a>(&'a self) -> TopNIterator<'a, T> {
        TopNIterator(self.data.iter())
    }
}

pub struct TopNIterator<'a, T>(std::collections::binary_heap::Iter<'a, ScoredItem<T>>);

impl<'a, T> Iterator for TopNIterator<'a, T> {
    type Item = &'a ScoredItem<T>;
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}

#[derive(Clone, PartialEq)]
pub struct SimObj<'ctx, 'b> {
    pub time: f64,
    pub count: usize,
    pub trs: TRS<'ctx, 'b>,
    pub obj_meta: f64,
    pub obj_trs: f64,
    pub obj_acc: f64,
    pub obj_gen: f64,
    pub ln_search_prior: f64,
    pub ln_search_likelihood: f64,
    pub ln_search_posterior: f64,
    pub ln_predict_prior: f64,
    pub ln_predict_likelihood: f64,
    pub ln_predict_posterior: f64,
}

impl<'ctx, 'b> SimObj<'ctx, 'b> {
    pub fn try_new(obj: &MCTSObj<'ctx>, mcts: &TRSMCTS<'ctx, 'b>) -> Option<Self> {
        obj.play(mcts).map(|trs| SimObj {
            time: obj.time,
            count: obj.count,
            trs,
            obj_meta: obj.obj_meta,
            obj_trs: obj.obj_trs,
            obj_acc: obj.obj_acc,
            obj_gen: obj.obj_gen,
            ln_search_prior: obj.ln_search_prior,
            ln_search_likelihood: obj.ln_search_likelihood,
            ln_search_posterior: obj.ln_search_posterior,
            ln_predict_prior: obj.ln_predict_prior,
            ln_predict_likelihood: obj.ln_predict_likelihood,
            ln_predict_posterior: obj.ln_predict_posterior,
        })
    }
    /// Note: This method explicitly ignores any change in the meta-prior that might occur with new trials.
    pub fn update_posterior(&mut self, mcts: &TRSMCTS<'ctx, 'b>) {
        // Get the new likelihoods.
        let mut l1 = mcts.model.likelihood;
        l1.single = SingleLikelihood::Generalization(0.001);
        let mut l2 = mcts.model.likelihood;
        l2.single = SingleLikelihood::Generalization(0.0);
        let soft_generalization_likelihood = self.trs.log_likelihood(mcts.data, l1);
        self.obj_gen = self.trs.log_likelihood(mcts.data, l2);
        self.obj_acc = self.trs.log_likelihood(mcts.data, mcts.model.likelihood);

        // update the posterior values.
        self.ln_search_likelihood = self.obj_acc + soft_generalization_likelihood;
        self.ln_search_posterior = self.ln_search_prior * mcts.model.p_temp
            + self.ln_search_likelihood * mcts.model.l_temp;
        // After HL finds a meta-program, it doesn't care how it found it.
        self.ln_predict_likelihood = self.obj_acc + self.obj_gen;
        self.ln_predict_posterior = self.ln_predict_prior * mcts.model.p_temp
            + self.ln_predict_likelihood * mcts.model.l_temp;
    }
}

impl<'ctx, 'b> Eq for SimObj<'ctx, 'b> {}

#[derive(Deserialize)]
pub struct Args {
    pub arg_params: String,
    pub arg_data: String,
    pub arg_out: String,
    pub arg_best: String,
    pub arg_prediction: String,
    pub arg_all: String,
    pub arg_run: usize,
}

#[derive(Deserialize, Clone)]
pub struct Params {
    pub simulation: SimulationParams,
    pub mcts: MCTSParams,
    pub model: ModelParams,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct SimulationParams {
    pub timeout: usize,
    pub n_predictions: usize,
    pub confidence: f64,
    pub deterministic: bool,
    pub lo: usize,
    pub hi: usize,
    pub signature: String,
    pub background: String,
    pub top_n: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Routine {
    #[serde(rename = "type")]
    pub tp: RoutineType,
    pub examples: Vec<Datum>,
    pub name: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Problem {
    pub id: String,
    pub program: String,
    pub data: Vec<Datum>,
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

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Value {
    Int(usize),
    IntList(Vec<usize>),
    Bool(bool),
}
impl Value {
    pub fn to_term(&self, lex: &Lexicon, lhs: Option<Operator>) -> Result<Term, ()> {
        let base_term = match self {
            Value::Int(x) => Value::num_to_term(lex, *x)?,
            Value::IntList(xs) => Value::list_to_term(lex, &xs)?,
            Value::Bool(true) => Term::Application {
                op: lex.has_operator(Some("true"), 0).map_err(|_| ())?,
                args: vec![],
            },
            Value::Bool(false) => Term::Application {
                op: lex.has_operator(Some("false"), 0).map_err(|_| ())?,
                args: vec![],
            },
        };
        if let Some(op) = lhs {
            let op_term = Term::Application { op, args: vec![] };
            Ok(Term::Application {
                op: lex.has_operator(Some("."), 2).map_err(|_| ())?,
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
        let nil = lex.has_operator(Some("NIL"), 0).map_err(|_| ())?;
        let cons = lex.has_operator(Some("CONS"), 0).map_err(|_| ())?;
        let app = lex.has_operator(Some("."), 2).map_err(|_| ())?;
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
            op: lex.has_operator(Some("DIGIT"), 0).map_err(|_| ())?,
            args: vec![],
        };
        let num_const = Term::Application {
            op: lex.has_operator(Some(&n.to_string()), 0).map_err(|_| ())?,
            args: vec![],
        };
        Ok(Term::Application {
            op: lex.has_operator(Some("."), 2).map_err(|_| ())?,
            args: vec![digit_const, num_const],
        })
    }
    fn num_to_term(lex: &Lexicon, num: usize) -> Result<Term, ()> {
        match num {
            0..=9 => Value::make_digit(lex, num),
            _ => {
                let app = lex.has_operator(Some("."), 2).map_err(|_| ())?;
                let decc_term = Term::Application {
                    op: lex.has_operator(Some("DECC"), 0).map_err(|_| ())?,
                    args: vec![],
                };
                let num_term = Value::num_to_term(lex, num / 10)?;
                let inner_term = Term::Application {
                    op: app,
                    args: vec![decc_term, num_term],
                };
                let digit_term = Term::Application {
                    op: lex
                        .has_operator(Some(&(num % 10).to_string()), 0)
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
