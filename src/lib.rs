use itertools::Itertools;
use polytype::atype::TypeContext;
use programinduction::trs::{
    mcts::{MCTSObj, MCTSParams, TRSMCTS},
    metaprogram::{
        MetaProgram, MetaProgramControl, MetaProgramHypothesis, Move, State, StateLabel,
    },
    parse_lexicon, parse_rulecontexts, parse_rules, parse_trs, Datum as TRSDatum, Lexicon,
    ModelParams, SingleLikelihood, TRS,
};
use rand::Rng;
use serde_derive::{Deserialize, Serialize};
use std::{
    cmp::Ordering, collections::BinaryHeap, fs::read_to_string, io::Write, ops::Deref,
    path::PathBuf, process::exit,
};
use term_rewriting::{trace::Trace, NumberRepresentation, Operator, Rule, RuleContext, Term};

pub type Prediction = (usize, String);

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

pub fn init_csv_fd(filename: &str, header: &str) -> Result<std::fs::File, String> {
    let mut fd = str_err(std::fs::File::create(filename))?;
    str_err(writeln!(fd, "{}", header))?;
    Ok(fd)
    //"problem,run,order,trial,time,count,lmeta,ltrs,lgen,lacc,lposterior,accuracy,trs,metaprogram"
}

pub fn try_trs<'ctx, 'b>(
    trs: &str,
    lex: &mut Lexicon<'ctx, 'b>,
    background: &'b [Rule],
    params: &Params,
    borrowed_data: &[&TRSDatum],
    examples: &[Rule],
) {
    let mut trs = parse_trs(trs, lex, true, background).unwrap();
    trs.set_bounds(params.simulation.lo, params.simulation.hi);
    let lp = trs.log_prior(params.model.prior);
    let ll = trs.log_likelihood(borrowed_data, params.model.likelihood);
    println!(
        "{:.4},{:.4},{:.4},\"{}\"",
        lp,
        ll,
        lp / 2.0 + ll,
        format!("{}", trs).lines().join(" ")
    );
    for datum in examples {
        let term = make_prediction(&trs, &datum.lhs, params);
        println!(
            "{} => {}\n",
            datum.pretty(lex.signature()),
            term.pretty(lex.signature())
        );
    }
}

pub fn try_program<'ctx, 'b>(
    t: TRS<'ctx, 'b>,
    moves: Vec<Move<'ctx>>,
    ctl: MetaProgramControl<'b>,
) -> Vec<MetaProgramHypothesis<'ctx, 'b>> {
    let mp = MetaProgram::new(t, vec![]);
    let h = State::from_meta(&mp, ctl);
    let mut stack = vec![h];
    for mv in moves {
        println!("# {}", mv);
        let mut processed = vec![];
        while let Some(mut h) = stack.pop() {
            if h.label != StateLabel::Failed {
                if h.spec.is_none() {
                    let these_moves = h.available_moves(ctl);
                    if let Some(w) = these_moves.iter().find(|(imv, _)| *imv == mv).map(|x| x.1) {
                        let z: f64 = these_moves.iter().map(|x| x.1).sum();
                        h.make_move(&mv, w / z, ctl.data);
                        if h.spec.is_none() {
                            processed.push(h);
                        } else {
                            stack.push(h);
                        }
                    } else {
                        h.label = StateLabel::Failed;
                    }
                } else {
                    let these_moves = h.available_moves(ctl);
                    let z: f64 = these_moves.iter().map(|x| x.1).sum();
                    for (mv, w) in h.available_moves(ctl) {
                        let mut new_h = h.clone();
                        new_h.make_move(&mv, w / z, ctl.data);
                        if new_h.spec.is_none() {
                            processed.push(new_h);
                        } else {
                            stack.push(new_h);
                        }
                    }
                }
            }
        }
        stack = processed;
    }
    println!("# processing {} hypotheses", stack.len());
    stack
        .into_iter()
        .enumerate()
        .inspect(|(i, h)| println!("# {}, {}", i, h.path))
        .filter_map(|(_, s)| Some(MetaProgramHypothesis::new(ctl, s.metaprogram()?)))
        .collect_vec()
}

pub fn process_prediction<'ctx, 'b>(query: &Rule, best: &TRS<'ctx, 'b>, params: &Params) -> bool {
    query.rhs[0] == make_prediction(best, &query.lhs, params)
}

fn make_prediction<'a, 'b>(trs: &TRS<'a, 'b>, input: &Term, params: &Params) -> Term {
    let utrs = trs.full_utrs();
    let lex = trs.lexicon();
    let sig = lex.signature();
    let patterns = utrs.patterns(sig);
    let trace = Trace::new(
        &utrs,
        sig,
        input,
        params.model.likelihood.p_observe,
        params.model.likelihood.max_steps,
        params.model.likelihood.max_size,
        &patterns,
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
        if self.data.len() < self.size && self.data.iter().all(|x| x.data.key() != datum.data.key())
        {
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
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
    pub fn len(&self) -> usize {
        self.data.len()
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
    pub fn update_posterior(&mut self, ctl: &MetaProgramControl<'b>) {
        // Get the new likelihoods.
        let mut l2 = ctl.model.likelihood;
        l2.single = SingleLikelihood::Generalization(0.0);
        self.hyp.ln_wf = self.trs.log_likelihood(ctl.data, l2);
        self.hyp.ln_acc = self.trs.log_likelihood(ctl.data, ctl.model.likelihood);

        // After HL finds a meta-program, it doesn't care how it found it.
        self.hyp.ln_likelihood = self.hyp.ln_acc + self.hyp.ln_wf;
        self.hyp.ln_posterior =
            self.hyp.ln_prior * ctl.model.p_temp + self.hyp.ln_likelihood * ctl.model.l_temp;
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
    pub verbose: bool,
    pub p_refactor: f64,
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
    pub fn to_rule(
        &self,
        lex: &Lexicon,
        concept: Operator,
        rep: NumberRepresentation,
    ) -> Result<Rule, ()> {
        let lhs = self.i.to_term(lex, Some(concept), rep)?;
        let rhs = self.o.to_term(lex, None, rep)?;
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
    pub fn to_term(
        &self,
        lex: &Lexicon,
        lhs: Option<Operator>,
        rep: NumberRepresentation,
    ) -> Result<Term, ()> {
        let base_term = match self {
            Value::Int(x) => Value::num_to_term(lex, *x, rep)?,
            Value::IntList(xs) => Value::list_to_term(lex, &xs, rep)?,
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
            let args = vec![base_term];
            Ok(Term::Application { op, args })
        } else {
            Ok(base_term)
        }
    }
    fn list_to_term(lex: &Lexicon, xs: &[usize], rep: NumberRepresentation) -> Result<Term, ()> {
        let ts: Vec<Term> = xs
            .iter()
            .map(|&x| Value::num_to_term(lex, x, rep))
            .rev()
            .collect::<Result<Vec<_>, _>>()?;
        let nil = lex.has_operator(Some("NIL"), 0).map_err(|_| ())?;
        let cons = lex.has_operator(Some("CONS"), 2).map_err(|_| ())?;
        let mut term = Term::Application {
            op: nil,
            args: vec![],
        };
        for t in ts {
            term = Term::Application {
                op: cons,
                args: vec![t, term],
            };
        }
        Ok(term)
    }
    fn num_to_term(lex: &Lexicon, num: usize, rep: NumberRepresentation) -> Result<Term, ()> {
        Term::from_usize(num, lex.signature(), rep).ok_or(())
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

// pub fn search_batch<'ctx, 'b, R: Rng>(
//     lex: Lexicon<'ctx, 'b>,
//     background: &'b [Rule],
//     examples: &'b [Rule],
//     train_set_size: usize,
//     params: &mut Params,
//     (problem, order): (&str, usize),
//     best_filename: &str,
//     rng: &mut R,
// ) -> Result<f64, String> {
//     let now = Instant::now();
//     let mut best_file = std::fs::File::create(best_filename).expect("file");
//     let mut top_n: TopN<Box<MetaProgramHypothesisWrapper>> = TopN::new(params.simulation.top_n);
//     // TODO: hacked in constants.
//     let mpctl1 = MetaProgramControl::new(
//         &[],
//         &params.model,
//         LearningMode::Refactor,
//         params.mcts.atom_weights,
//         7,
//         50,
//     );
//     let mpctl2 = MetaProgramControl::new(
//         &[],
//         &params.model,
//         LearningMode::Sample,
//         params.mcts.atom_weights,
//         2,
//         22,
//     );
//     let timeout = params.simulation.timeout;
//     let train_data = convert_examples_to_data(&examples[..train_set_size]);
//     let borrowed_data = train_data.iter().collect_vec();
//     let mut t0 = TRS::new_unchecked(&lex, params.simulation.deterministic, background, vec![]);
//     t0.set_bounds(params.simulation.lo, params.simulation.hi);
//     t0.identify_symbols();
//     let p0 = MetaProgram::from(t0);
//     // No need to compute posterior. `chain` will do that for us.
//     let h01 = MetaProgramHypothesis::new(mpctl1, &p0);
//     let h02 = MetaProgramHypothesis::new(mpctl2, &p0);
//     let mut ctl1 = Control::new(0, timeout * 1000, 0, 0, 0);
//     let mut ctl2 = Control::new(0, timeout * 1000, 0, 0, 0);
//     let swap = 5000;
//     //let ladder = TemperatureLadder(vec![
//     //    Temperature::new(temp(0, 5, 12), temp(0, 5, 12)),
//     //    Temperature::new(temp(1, 5, 12), temp(1, 5, 12)),
//     //    Temperature::new(temp(2, 5, 12), temp(2, 5, 12)),
//     //    Temperature::new(temp(3, 5, 12), temp(3, 5, 12)),
//     //    Temperature::new(temp(4, 5, 12), temp(4, 5, 12)),
//     //]);
//     let ladder = TemperatureLadder(vec![Temperature::new(2.0, 1.0)]);
//     //let ladder = TemperatureLadder(vec![Temperature::new(2.0 * temp(4, 5, 12), temp(4, 5, 12))]);
//     let mut chain1 = ParallelTempering::new(h01, &borrowed_data, ladder.clone(), swap, rng);
//     let mut chain2 = ParallelTempering::new(h02, &borrowed_data, ladder, swap, rng);
//     update_data_mcmc(
//         &mut chain1,
//         &mut top_n,
//         &borrowed_data,
//         params.simulation.top_n,
//     );
//     update_data_mcmc(
//         &mut chain2,
//         &mut top_n,
//         &borrowed_data,
//         params.simulation.top_n,
//     );
//     let mut best = std::f64::INFINITY;
//     {
//         println!("# drawing samples: {}ms", now.elapsed().as_millis());
//         while let (Some(sample1), Some(sample2)) = (
//             chain1.internal_next(&mut ctl1, rng),
//             chain2.internal_next(&mut ctl2, rng),
//         ) {
//             print_hypothesis_mcmc(problem, order, 1, train_set_size, &sample1, false, None);
//             let score = -sample1.at_temperature(Temperature::new(4.0, 1.0));
//             if score < best {
//                 writeln!(
//                     &mut best_file,
//                     "{}",
//                     hypothesis_string_mcmc(problem, order, 1, train_set_size, &sample1, true, None)
//                 )
//                 .expect("written");
//                 best = score;
//             }
//             top_n.add(ScoredItem {
//                 score,
//                 data: Box::new(MetaProgramHypothesisWrapper(sample1.clone())),
//             });
//             print_hypothesis_mcmc(problem, order, 1, train_set_size, &sample2, false, None);
//             let score = -sample2.at_temperature(Temperature::new(4.0, 1.0));
//             if score < best {
//                 writeln!(
//                     &mut best_file,
//                     "{}",
//                     hypothesis_string_mcmc(problem, order, 1, train_set_size, &sample2, true, None)
//                 )
//                 .expect("written");
//                 best = score;
//             }
//             top_n.add(ScoredItem {
//                 score,
//                 data: Box::new(MetaProgramHypothesisWrapper(sample2.clone())),
//             });
//         }
//     }
//     let mut n_correct = 0;
//     let mut n_tried = 0;
//     let best = &top_n.least().unwrap().data.0.state;
//     for query in &examples[train_set_size..] {
//         let correct = process_prediction(query, &best.trs, params);
//         n_correct += correct as usize;
//         n_tried += 1;
//     }
//     println!("# END OF SEARCH");
//     println!("# top hypotheses:");
//     top_n.iter().sorted().enumerate().rev().for_each(|(i, h)| {
//         println!(
//             "# {}\t{}",
//             i,
//             hypothesis_string_mcmc(problem, order, 1, train_set_size, &h.data.0, false, None)
//         )
//     });
//     println!("#");
//     println!("# problem: {}", problem);
//     println!("# order: {}", order);
//     println!("# samples 1: {:?}", chain1.samples());
//     println!("# samples 2: {:?}", chain2.samples());
//     println!("# acceptance ratio 1: {:?}", chain1.acceptance_ratio());
//     println!("# acceptance ratio 2: {:?}", chain2.acceptance_ratio());
//     println!("# swap ratio 1: {:?}", chain1.swaps());
//     println!("# swap ratio 2: {:?}", chain2.swaps());
//     println!("# best hypothesis metaprogram: {}", best.path);
//     println!(
//         "# best hypothesis TRS: {}",
//         best.trs.to_string().lines().join(" ")
//     );
//     println!("# correct predictions rational: {}/{}", n_correct, n_tried);
//     // TODO: fix search time
//     Ok(0.0)
// }
