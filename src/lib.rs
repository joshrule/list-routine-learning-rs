use itertools::Itertools;
use polytype::atype::TypeContext;
use programinduction::trs::{
    mcts::{MCTSObj, MCTSParams, Move, TRSMCTS},
    parse_lexicon, parse_rulecontexts, parse_rules, parse_trs, Lexicon, ModelParams,
    SingleLikelihood, TRS,
};
use rand::Rng;
use serde_derive::{Deserialize, Serialize};
use std::{collections::BinaryHeap, fs::read_to_string, ops::Deref, path::PathBuf, process::exit};
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

pub trait Keyed {
    type Key: Eq;
    fn key(&self) -> &Self::Key;
}

impl<T: Keyed> Keyed for Box<T> {
    type Key = T::Key;
    fn key(&self) -> &Self::Key {
        self.deref().key()
    }
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

impl<T: Eq + Keyed> TopN<T> {
    pub fn new(size: usize) -> Self {
        TopN {
            size,
            data: BinaryHeap::with_capacity(size),
        }
    }
    pub fn add(&mut self, datum: ScoredItem<T>) {
        if self.data.len() < self.size {
            self.data.push(datum);
        } else if let Some(worst) = self.most() {
            if datum.score < worst.score {
                if self.iter().all(|x| x.data.key() != datum.data.key()) {
                    self.data.pop();
                    self.data.push(datum);
                }
            }
        }
    }
    pub fn to_vec(self) -> Vec<ScoredItem<T>> {
        self.data.into_sorted_vec()
    }
    pub fn most(&self) -> Option<&ScoredItem<T>> {
        self.data.peek()
    }
    pub fn least(&self) -> Option<&ScoredItem<T>> {
        self.data.iter().min()
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
    pub hyp: MCTSObj<'ctx>,
    pub trs: TRS<'ctx, 'b>,
}

impl<'ctx, 'b> SimObj<'ctx, 'b> {
    pub fn try_new(hyp: &MCTSObj<'ctx>, mcts: &TRSMCTS<'ctx, 'b>) -> Option<Self> {
        hyp.play(mcts).map(|trs| SimObj {
            hyp: hyp.clone(),
            trs,
        })
    }
    /// Note: This method explicitly ignores any change in the meta-prior that might occur with new trials.
    pub fn update_posterior(&mut self, mcts: &TRSMCTS<'ctx, 'b>) {
        // Get the new likelihoods.
        let mut l2 = mcts.model.likelihood;
        l2.single = SingleLikelihood::Generalization(0.0);
        self.hyp.ln_wf = self.trs.log_likelihood(mcts.data, l2);
        self.hyp.ln_acc = self.trs.log_likelihood(mcts.data, mcts.model.likelihood);

        // After HL finds a meta-program, it doesn't care how it found it.
        self.hyp.ln_likelihood = self.hyp.ln_acc + self.hyp.ln_wf;
        self.hyp.ln_posterior =
            self.hyp.ln_prior * mcts.model.p_temp + self.hyp.ln_likelihood * mcts.model.l_temp;
    }
}

impl<'ctx, 'b> Eq for SimObj<'ctx, 'b> {}

impl<'ctx, 'b> Keyed for SimObj<'ctx, 'b> {
    type Key = Vec<Move<'ctx>>;
    fn key(&self) -> &Self::Key {
        &self.hyp.moves
    }
}

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

#[cfg(test)]
mod tests {
    use crate::*;

    #[derive(PartialEq, Eq, PartialOrd, Ord, Debug)]
    struct Int(usize);

    impl Keyed for Int {
        type Key = usize;
        fn key(&self) -> &Self::Key {
            &self.0
        }
    }

    fn make_item(n: usize) -> ScoredItem<Int> {
        ScoredItem {
            data: Int(n),
            score: n as f64,
        }
    }

    #[test]
    fn top_n_test() {
        let mut top_n: TopN<Int> = TopN::new(5);

        // It initializes correctly.
        assert_eq!(top_n.size, 5);
        assert_eq!(top_n.data.len(), 0);

        // It can add an item.
        top_n.add(make_item(6));
        assert_eq!(top_n.data.len(), 1);
        assert_eq!(top_n.most().unwrap().data, Int(6));
        assert_eq!(top_n.least().unwrap().data, Int(6));

        // It correctly updates for multiple items.
        top_n.add(make_item(3));
        top_n.add(make_item(8));
        top_n.add(make_item(1));
        top_n.add(make_item(4));

        assert_eq!(top_n.data.len(), 5);
        assert_eq!(top_n.most().unwrap().data.0, 8);
        assert_eq!(top_n.least().unwrap().data.0, 1);

        // It correctly handles overflow.
        top_n.add(make_item(9));

        assert_eq!(top_n.data.len(), 5);
        assert_eq!(top_n.most().unwrap().data.0, 8);
        assert_eq!(top_n.least().unwrap().data.0, 1);

        top_n.add(make_item(7));

        assert_eq!(top_n.data.len(), 5);
        assert_eq!(top_n.most().unwrap().data.0, 7);
        assert_eq!(top_n.least().unwrap().data.0, 1);

        top_n.add(make_item(2));

        assert_eq!(top_n.data.len(), 5);
        assert_eq!(top_n.most().unwrap().data.0, 6);
        assert_eq!(top_n.least().unwrap().data.0, 1);

        top_n.add(make_item(0));

        assert_eq!(top_n.data.len(), 5);
        assert_eq!(top_n.most().unwrap().data.0, 4);
        assert_eq!(top_n.least().unwrap().data.0, 0);

        top_n.add(make_item(2));

        // We have the expected items at close.
        let data = top_n
            .to_vec()
            .iter()
            .sorted()
            .map(|x| x.data.0)
            .collect::<Vec<_>>();
        assert_eq!(data, vec![0, 1, 2, 2, 3]);
    }
}
