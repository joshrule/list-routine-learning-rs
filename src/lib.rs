use docopt::Docopt;
use itertools::Itertools;
use polytype::atype::TypeContext;
use programinduction::{
    hypotheses::{Bayesable, Temperable},
    inference::{Control, ParallelTempering, TemperatureLadder},
    trs::{
        mcts::{MCTSObj, TRSMCTS},
        metaprogram::{
            LearningMode, MetaProgram, MetaProgramControl, MetaProgramHypothesis, Move, State,
            StateLabel, Temperature,
        },
        parse_lexicon, parse_rulecontexts, parse_rules, parse_trs, Datum as TRSDatum, Lesion,
        Lexicon, ModelParams, SingleLikelihood, TRS,
    },
};
use rand::{
    distributions::{Distribution, WeightedIndex},
    Rng,
};
use serde_derive::{Deserialize, Serialize};
use std::{
    cmp::Ordering,
    collections::BinaryHeap,
    f64,
    fs::read_to_string,
    fs::File,
    io::{BufReader, Write},
    ops::Deref,
    path::PathBuf,
    process::exit,
    str,
    time::Instant,
};
use term_rewriting::{
    trace::{Trace, TraceState},
    NumberRepresentation, Operator, Rule, RuleContext, Term,
};

pub type Prediction = (usize, String);

#[derive(Clone, PartialEq, Eq)]
pub struct MetaProgramHypothesisWrapper<'ctx, 'b>(MetaProgramHypothesis<'ctx, 'b>);

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SimulationMode {
    Online,
    Batch,
}

impl<'ctx, 'b> Keyed for MetaProgramHypothesisWrapper<'ctx, 'b> {
    type Key = MetaProgram<'ctx, 'b>;
    fn key(&self) -> &Self::Key {
        &self.0.state.path
    }
}

pub fn temp(i: usize, n: usize, max_t: usize) -> f64 {
    (i as f64 * (max_t as f64).ln() / ((n - 1) as f64)).exp()
}

pub fn load_args() -> Result<(Params, usize, String, String, String, String, String), String> {
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

pub fn load_problem(data_filename: &str) -> Result<(Vec<Datum>, String), String> {
    let path: PathBuf = PathBuf::from(data_filename);
    let file = str_err(File::open(path))?;
    let reader = BufReader::new(file);
    let problem: Problem = str_err(serde_json::from_reader(reader))?;
    Ok((problem.data, problem.id))
}

pub fn identify_concept(lex: &Lexicon) -> Result<Operator, String> {
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
    let mut trs = parse_trs(trs, lex, params.simulation.deterministic, background).unwrap();
    trs.set_bounds(params.simulation.lo, params.simulation.hi);
    let lp = trs.log_prior(params.model.prior);
    let ll = trs.log_likelihood(borrowed_data, params.model.likelihood);
    println!(
        "{:.4},{:.4},{:.4},\"{}\"",
        lp,
        ll,
        lp + ll,
        format!("{}", trs).lines().join(" ")
    );
    for datum in examples {
        let (prob, outs) = includes_output(&trs, lex, &datum.lhs, &datum.rhs().unwrap(), params);
        println!(
            "{} => {:.4} ({})\n    {}",
            datum.pretty(lex.signature()),
            prob,
            outs.len(),
            outs.iter()
                .map(|(t, p)| format!("({}, {:.4})", t.pretty(lex.signature()), p))
                .format("\n    ")
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

pub fn includes_output<'a, 'b>(
    trs: &TRS<'a, 'b>,
    lex: &mut Lexicon<'a, 'b>,
    input: &Term,
    output: &Term,
    params: &Params,
) -> (f64, Vec<(Term, f64)>) {
    let utrs = trs.full_utrs();
    let sig = trs.lexicon().signature();
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
    println!("mass: {}", trace.mass());
    println!("depth: {}", trace.depth());
    println!("size: {}", trace.size());
    for (i, n) in trace.iter().enumerate() {
        let nd = &trace[n];
        println!(
            "{}. {} {:.4} {} {} {}",
            i,
            nd.term().pretty(lex.signature()),
            nd.log_p(),
            nd.depth(),
            nd.state(),
            nd.is_leaf(),
        )
    }
    let outputs = trace
        .iter()
        .filter(|n| trace[*n].state() == TraceState::Normal)
        .sorted_by_key(|n| -(trace[*n].log_p() * 1e9) as usize)
        .map(|n| (trace[n].term().clone(), trace[n].log_p()))
        .collect_vec();
    let p = trace.rewrites_to(output, |correct, predicted| {
        if correct == predicted {
            0.0
        } else {
            std::f64::NEG_INFINITY
        }
    });
    (p, outputs)
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
        self.hyp.ln_posterior = self.hyp.ln_prior + self.hyp.ln_likelihood;
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
    pub model: ModelParams,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct SimulationParams {
    pub timeout: usize,
    pub steps: usize,
    pub n_predictions: usize,
    pub deterministic: bool,
    pub lo: usize,
    pub hi: usize,
    pub signature: String,
    pub background: String,
    pub top_n: usize,
    pub verbose: bool,
    pub p_refactor: f64,
    pub mode: SimulationMode,
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

pub fn search_online<'ctx, 'b, R: Rng>(
    lex: Lexicon<'ctx, 'b>,
    background: &'b [Rule],
    examples: &'b [Rule],
    n_trials: usize,
    params: &mut Params,
    (problem, order, run): (&str, usize, usize),
    best_file: &mut File,
    prediction_file: &mut File,
    reservoir: &mut Reservoir<String>,
    rng: &mut R,
) -> Result<f64, String> {
    let now = Instant::now();
    let prior_temp = if matches!(params.model.lesion, Lesion::None) {
        2.0
    } else {
        1.0
    };
    let mut search_time = 0.0;
    let mut top_n: TopN<Box<MetaProgramHypothesisWrapper>> = TopN::new(params.simulation.top_n);
    let timeout = params.simulation.timeout;
    let steps = params.simulation.steps;
    // TODO: hacked in constants.
    let mpctl1 = MetaProgramControl::new(&[], &params.model, LearningMode::Refactor, 5.0, 7, 66);
    let mpctl2 = MetaProgramControl::new(&[], &params.model, LearningMode::Sample, 5.0, 2, 66);
    let mut t0 = TRS::new_unchecked(&lex, params.simulation.deterministic, background, vec![]);
    t0.set_bounds(params.simulation.lo, params.simulation.hi);
    t0.identify_symbols();
    let p0 = MetaProgram::from(t0);
    let h01 = MetaProgramHypothesis::new(mpctl1, &p0);
    let h02 = MetaProgramHypothesis::new(mpctl2, &p0);
    let mut ctl = Control::new(0, 0, 0, 0, 0);
    let swap = 25000;
    let ladder = TemperatureLadder(vec![
        Temperature::new(prior_temp, temp(0, 5, 1)),
        Temperature::new(prior_temp, temp(1, 5, 1)),
        Temperature::new(prior_temp, temp(2, 5, 1)),
        Temperature::new(prior_temp, temp(3, 5, 1)),
        Temperature::new(prior_temp, temp(4, 5, 1)),
    ]);
    // All of the computational work pre-search is here.
    let mut chain1 = ParallelTempering::new(h01, &[], ladder.clone(), swap, rng);
    for (i, chain) in chain1.pool.iter_mut().enumerate() {
        chain.1.current_mut().birth.count = i;
        if params.simulation.verbose {
            print_hypothesis_mcmc(
                problem,
                order,
                run,
                0,
                chain.1.current(),
                true,
                None,
                prior_temp,
            );
        }
        let score = -chain
            .1
            .current()
            .at_temperature(Temperature::new(prior_temp, 1.0));
        reservoir.add(
            || {
                hypothesis_string_mcmc(
                    problem,
                    order,
                    run,
                    0,
                    chain.1.current(),
                    true,
                    None,
                    prior_temp,
                )
            },
            rng,
        );
        top_n.add(ScoredItem {
            score,
            data: Box::new(MetaProgramHypothesisWrapper(chain.1.current().clone())),
        });
    }
    let mut chain2 = ParallelTempering::new(h02, &[], ladder, swap, rng);
    for (i, chain) in chain2.pool.iter_mut().enumerate() {
        chain.1.current_mut().birth.count = i + chain1.pool.len();
        if params.simulation.verbose {
            print_hypothesis_mcmc(
                problem,
                order,
                run,
                0,
                chain.1.current(),
                true,
                None,
                prior_temp,
            );
        }
        let score = -chain
            .1
            .current()
            .at_temperature(Temperature::new(prior_temp, 1.0));
        reservoir.add(
            || {
                hypothesis_string_mcmc(
                    problem,
                    order,
                    run,
                    0,
                    chain.1.current(),
                    true,
                    None,
                    prior_temp,
                )
            },
            rng,
        );
        top_n.add(ScoredItem {
            score,
            data: Box::new(MetaProgramHypothesisWrapper(chain.1.current().clone())),
        });
    }
    ctl.done_steps += chain1.pool.len() + chain2.pool.len();
    ctl.accepted = ctl.done_steps;
    let mut best;
    let data = (1..=n_trials)
        .map(|n| convert_examples_to_data(&examples[..n]))
        .collect_vec();
    let borrowed_data = data.iter().map(|x| x.iter().collect_vec()).collect_vec();
    let dist = WeightedIndex::new(&[
        params.simulation.p_refactor,
        1.0 - params.simulation.p_refactor,
    ])
    .unwrap();
    for n_data in 0..n_trials {
        println!("#   starting trial {}", n_data);
        update_data_mcmc(
            &mut chain1,
            &mut top_n,
            LearningMode::Refactor,
            prior_temp,
            &borrowed_data[n_data],
            params.simulation.top_n,
        );
        update_data_mcmc(
            &mut chain2,
            &mut top_n,
            LearningMode::Sample,
            prior_temp,
            &borrowed_data[n_data],
            params.simulation.top_n,
        );
        let h_initial = &top_n.least().unwrap().data.0;
        best = -h_initial.at_temperature(Temperature::new(prior_temp, 1.0));
        writeln!(
            best_file,
            "{}",
            hypothesis_string_mcmc(problem, order, run, n_data, h_initial, true, None, prior_temp)
        )
        .expect("written");
        for (i, chain) in chain1.pool.iter_mut().enumerate() {
            chain
                .1
                .set_temperature(Temperature::new(prior_temp, temp(i, 5, n_data + 1)));
        }
        for (i, chain) in chain2.pool.iter_mut().enumerate() {
            chain
                .1
                .set_temperature(Temperature::new(prior_temp, temp(i, 5, n_data + 1)));
        }
        ctl.extend_runtime(timeout * 1000);
        ctl.extend_steps(steps);
        let mut active_1 = params.simulation.p_refactor > 0.0;
        let mut active_2 = (1.0 - params.simulation.p_refactor) > 0.0;
        let trial_start = Instant::now();
        while active_1 || active_2 {
            // pick the thing to sample from
            let idx = dist.sample(rng);
            // Sample from it
            let sample = if idx == 0 {
                chain1.internal_next(&mut ctl, rng)
            } else {
                chain2.internal_next(&mut ctl, rng)
            };
            // Process the sample
            match sample {
                None => {
                    if idx == 0 {
                        active_1 = false;
                    } else {
                        active_2 = false;
                    }
                }
                Some(sample) => {
                    if params.simulation.verbose {
                        print_hypothesis_mcmc(
                            problem, order, run, n_data, sample, true, None, prior_temp,
                        );
                    }
                    let score = -sample.at_temperature(Temperature::new(prior_temp, 1.0));
                    if score < best {
                        writeln!(
                            best_file,
                            "{}",
                            hypothesis_string_mcmc(
                                problem, order, run, n_data, sample, true, None, prior_temp
                            )
                        )
                        .expect("written");
                        best = score;
                    }
                    reservoir.add(
                        || {
                            hypothesis_string_mcmc(
                                problem, order, run, n_data, sample, true, None, prior_temp,
                            )
                        },
                        rng,
                    );
                    top_n.add(ScoredItem {
                        score,
                        data: Box::new(MetaProgramHypothesisWrapper(sample.clone())),
                    });
                }
            }
            active_1 = active_1 && ctl.running();
            active_2 = active_2 && ctl.running();
        }
        search_time += trial_start.elapsed().as_secs_f64();
        let h_best = &top_n.least().unwrap().data.0;
        let query = &examples[n_data];
        let correct = process_prediction(query, &h_best.state.trs, params);
        writeln!(
            prediction_file,
            "{}",
            hypothesis_string_mcmc(
                problem,
                order,
                run,
                n_data,
                &h_best,
                true,
                Some(correct as usize as f64),
                prior_temp
            )
        )
        .ok();
        if n_data + 1 == n_trials {
            summarize_search(
                &top_n,
                &chain1,
                &chain2,
                problem,
                order,
                run,
                n_data,
                n_trials,
                search_time,
                now.elapsed().as_secs_f64(),
                prior_temp,
            );
        }
    }
    Ok(search_time)
}

pub fn search_batch<'ctx, 'b, R: Rng>(
    lex: Lexicon<'ctx, 'b>,
    background: &'b [Rule],
    examples: &'b [Rule],
    n_data: usize,
    params: &mut Params,
    (problem, order, run): (&str, usize, usize),
    best_file: &mut File,
    prediction_file: &mut File,
    reservoir: &mut Reservoir<String>,
    rng: &mut R,
) -> Result<f64, String> {
    let now = Instant::now();
    let prior_temp = if matches!(params.model.lesion, Lesion::None) {
        2.0
    } else {
        1.0
    };
    let mut top_n: TopN<Box<MetaProgramHypothesisWrapper>> = TopN::new(params.simulation.top_n);
    let timeout = params.simulation.timeout;
    let steps = params.simulation.steps;
    // TODO: hacked in constants.
    let mpctl1 = MetaProgramControl::new(&[], &params.model, LearningMode::Refactor, 5.0, 7, 66);
    let mpctl2 = MetaProgramControl::new(&[], &params.model, LearningMode::Sample, 5.0, 2, 66);
    let mut t0 = TRS::new_unchecked(&lex, params.simulation.deterministic, background, vec![]);
    t0.set_bounds(params.simulation.lo, params.simulation.hi);
    t0.identify_symbols();
    let p0 = MetaProgram::from(t0);
    let h01 = MetaProgramHypothesis::new(mpctl1, &p0);
    let h02 = MetaProgramHypothesis::new(mpctl2, &p0);
    let mut ctl = Control::new(steps, timeout * 1000, 0, 0, 0);
    let swap = 25000;
    let ladder = TemperatureLadder(vec![
        Temperature::new(prior_temp, temp(0, 5, n_data + 1)),
        Temperature::new(prior_temp, temp(1, 5, n_data + 1)),
        Temperature::new(prior_temp, temp(2, 5, n_data + 1)),
        Temperature::new(prior_temp, temp(3, 5, n_data + 1)),
        Temperature::new(prior_temp, temp(4, 5, n_data + 1)),
    ]);
    // All of the computational work pre-search is here.
    let mut chain1 = ParallelTempering::new(h01, &[], ladder.clone(), swap, rng);
    for (i, chain) in chain1.pool.iter_mut().enumerate() {
        chain.1.current_mut().birth.count = i;
        if params.simulation.verbose {
            print_hypothesis_mcmc(
                problem,
                order,
                run,
                0,
                chain.1.current(),
                true,
                None,
                prior_temp,
            );
        }
        let score = -chain
            .1
            .current()
            .at_temperature(Temperature::new(prior_temp, 1.0));
        reservoir.add(
            || {
                hypothesis_string_mcmc(
                    problem,
                    order,
                    run,
                    0,
                    chain.1.current(),
                    true,
                    None,
                    prior_temp,
                )
            },
            rng,
        );
        top_n.add(ScoredItem {
            score,
            data: Box::new(MetaProgramHypothesisWrapper(chain.1.current().clone())),
        });
    }
    let mut chain2 = ParallelTempering::new(h02, &[], ladder, swap, rng);
    for (i, chain) in chain2.pool.iter_mut().enumerate() {
        chain.1.current_mut().birth.count = i + chain1.pool.len();
        if params.simulation.verbose {
            print_hypothesis_mcmc(
                problem,
                order,
                run,
                0,
                chain.1.current(),
                true,
                None,
                prior_temp,
            );
        }
        let score = -chain
            .1
            .current()
            .at_temperature(Temperature::new(prior_temp, 1.0));
        reservoir.add(
            || {
                hypothesis_string_mcmc(
                    problem,
                    order,
                    run,
                    0,
                    chain.1.current(),
                    true,
                    None,
                    prior_temp,
                )
            },
            rng,
        );
        top_n.add(ScoredItem {
            score,
            data: Box::new(MetaProgramHypothesisWrapper(chain.1.current().clone())),
        });
    }
    ctl.done_steps += chain1.pool.len() + chain2.pool.len();
    ctl.accepted = ctl.done_steps;
    let data = convert_examples_to_data(&examples[..n_data]);
    let borrowed_data = data.iter().collect_vec();
    let dist = WeightedIndex::new(&[
        params.simulation.p_refactor,
        1.0 - params.simulation.p_refactor,
    ])
    .unwrap();
    update_data_mcmc(
        &mut chain1,
        &mut top_n,
        LearningMode::Refactor,
        prior_temp,
        &borrowed_data,
        params.simulation.top_n,
    );
    update_data_mcmc(
        &mut chain2,
        &mut top_n,
        LearningMode::Sample,
        prior_temp,
        &borrowed_data,
        params.simulation.top_n,
    );
    let h_initial = &top_n.least().unwrap().data.0;
    let mut best = -h_initial.at_temperature(Temperature::new(prior_temp, 1.0));
    writeln!(
        best_file,
        "{}",
        hypothesis_string_mcmc(problem, order, run, n_data, h_initial, true, None, prior_temp)
    )
    .expect("written");
    for (i, chain) in chain1.pool.iter_mut().enumerate() {
        chain
            .1
            .set_temperature(Temperature::new(prior_temp, temp(i, 5, n_data + 1)));
    }
    for (i, chain) in chain2.pool.iter_mut().enumerate() {
        chain
            .1
            .set_temperature(Temperature::new(prior_temp, temp(i, 5, n_data + 1)));
    }
    let mut active_1 = params.simulation.p_refactor > 0.0;
    let mut active_2 = (1.0 - params.simulation.p_refactor) > 0.0;
    let trial_start = Instant::now();
    while active_1 || active_2 {
        // pick the thing to sample from
        let idx = dist.sample(rng);
        // Sample from it
        let sample = if idx == 0 {
            chain1.internal_next(&mut ctl, rng)
        } else {
            chain2.internal_next(&mut ctl, rng)
        };
        // Process the sample
        match sample {
            None => {
                if idx == 0 {
                    active_1 = false;
                } else {
                    active_2 = false;
                }
            }
            Some(sample) => {
                if params.simulation.verbose {
                    print_hypothesis_mcmc(
                        problem, order, run, n_data, sample, true, None, prior_temp,
                    );
                }
                let score = -sample.at_temperature(Temperature::new(prior_temp, 1.0));
                if score < best {
                    writeln!(
                        best_file,
                        "{}",
                        hypothesis_string_mcmc(
                            problem, order, run, n_data, sample, true, None, prior_temp
                        )
                    )
                    .expect("written");
                    best = score;
                }
                reservoir.add(
                    || {
                        hypothesis_string_mcmc(
                            problem, order, run, n_data, sample, true, None, prior_temp,
                        )
                    },
                    rng,
                );
                top_n.add(ScoredItem {
                    score,
                    data: Box::new(MetaProgramHypothesisWrapper(sample.clone())),
                });
            }
        }
        active_1 = active_1 && ctl.running();
        active_2 = active_2 && ctl.running();
    }
    let search_time = trial_start.elapsed().as_secs_f64();
    summarize_search(
        &top_n,
        &chain1,
        &chain2,
        problem,
        order,
        run,
        n_data,
        1,
        search_time,
        now.elapsed().as_secs_f64(),
        prior_temp,
    );
    let mut n_correct = 0;
    let mut n_tried = 0;
    let h_best = &top_n.least().unwrap().data.0;
    for query in &examples[n_data..] {
        let correct = process_prediction(query, &h_best.state.trs, params);
        n_correct += correct as usize;
        n_tried += 1;
    }
    writeln!(
        prediction_file,
        "{}",
        hypothesis_string_mcmc(
            problem,
            order,
            run,
            n_data,
            &h_best,
            true,
            Some(n_correct as f64 / n_tried as f64),
            prior_temp
        )
    )
    .ok();
    println!("# correct predictions: {}/{}", n_correct, n_tried);
    // END LOOP
    Ok(search_time)
}

fn summarize_search<'a, 'b>(
    top_n: &TopN<Box<MetaProgramHypothesisWrapper<'a, 'b>>>,
    chain1: &ParallelTempering<MetaProgramHypothesis<'a, 'b>>,
    chain2: &ParallelTempering<MetaProgramHypothesis<'a, 'b>>,
    problem: &str,
    order: usize,
    run: usize,
    n_data: usize,
    n_trials: usize,
    search_time: f64,
    run_time: f64,
    prior_temp: f64,
) {
    let best = &top_n.least().unwrap().data.0.state;
    notice("ending search", 0);
    println!("#");
    notice("top hypotheses:", 0);
    top_n.iter().sorted().enumerate().rev().for_each(|(i, h)| {
        println!(
            "#   {}\t{}",
            i,
            hypothesis_string_mcmc(problem, order, run, n_data, &h.data.0, true, None, prior_temp)
        )
    });
    println!("#");
    println!("# problem: {}", problem);
    println!("# order: {}", order);
    println!("# samples 1: {:?}", chain1.samples());
    println!("# samples 2: {:?}", chain2.samples());
    println!("# acceptance ratio 1: {:?}", chain1.acceptance_ratio());
    println!("# acceptance ratio 2: {:?}", chain2.acceptance_ratio());
    println!("# swap ratio 1: {:?}", chain1.swaps());
    println!("# swap ratio 2: {:?}", chain2.swaps());
    println!("# best hypothesis metaprogram: {}", best.path);
    println!(
        "# best hypothesis TRS: {}",
        best.trs.to_string().lines().join(" ")
    );
    println!("# run time (s): {:.4}", run_time);
    println!("# search time (s): {:.4}", search_time);
    println!(
        "# mean search time / trial (s): {:.4}",
        search_time / (n_trials as f64)
    );
    println!(
        "# steps: {}",
        chain1
            .samples()
            .iter()
            .chain(&chain2.samples())
            .sum::<usize>()
    );
    println!(
        "# mean steps/sec: {:.4}",
        (chain1.samples().iter().sum::<usize>() as f64
            + chain2.samples().iter().sum::<usize>() as f64)
            / search_time
    );
}

pub fn convert_examples_to_data(examples: &[Rule]) -> Vec<TRSDatum> {
    examples
        .iter()
        .cloned()
        .enumerate()
        .map(|(i, e)| {
            if i < examples.len() - 1 {
                TRSDatum::Full(e)
            } else {
                TRSDatum::Partial(e.lhs)
            }
        })
        .collect_vec()
}

fn print_hypothesis_mcmc(
    problem: &str,
    order: usize,
    run: usize,
    trial: usize,
    h: &MetaProgramHypothesis,
    print_meta: bool,
    correct: Option<f64>,
    prior_temp: f64,
) {
    println!(
        "{}",
        hypothesis_string_mcmc(problem, order, run, trial, h, print_meta, correct, prior_temp)
    );
}

fn hypothesis_string_mcmc(
    problem: &str,
    order: usize,
    run: usize,
    trial: usize,
    h: &MetaProgramHypothesis,
    print_meta: bool,
    correct: Option<f64>,
    prior_temp: f64,
) -> String {
    hypothesis_string_inner(
        problem,
        order,
        run,
        trial,
        h.state.trs().unwrap(),
        &h.state.metaprogram().unwrap().iter().cloned().collect_vec(),
        h.birth.time,
        h.birth.count,
        &[
            h.ln_meta,
            h.ln_trs,
            h.ln_wf,
            h.ln_acc,
            h.at_temperature(Temperature::new(prior_temp, 1.0)),
        ],
        correct,
        print_meta,
    )
}

fn hypothesis_string_inner(
    problem: &str,
    order: usize,
    run: usize,
    trial: usize,
    trs: &TRS,
    moves: &[Move],
    time: usize,
    count: usize,
    objective: &[f64],
    correct: Option<f64>,
    print_meta: bool,
) -> String {
    //let trs_str = trs.to_string().lines().join(" ");
    let trs_len = trs.size();
    let objective_string = format!("{: >10.8}", objective.iter().format("\t"));
    let meta_string = format!("{}", moves.iter().format("."));
    let trs_string = format!("{}", trs).lines().join(" ");
    match (print_meta, correct) {
        (false, None) => {
            format!(
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t\"{}\"",
                problem, order, run, trial, trs_len, objective_string, count, time, trs_string
            )
        }
        (false, Some(c)) => {
            format!(
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t\"{}\"",
                problem, order, run, trial, trs_len, objective_string, count, time, c, trs_string
            )
        }
        (true, None) => {
            format!(
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t\"{}\"\t\"{}\"",
                problem,
                order,
                run,
                trial,
                trs_len,
                objective_string,
                count,
                time,
                trs_string,
                meta_string
            )
        }
        (true, Some(c)) => {
            format!(
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t\"{}\"\t\"{}\"",
                problem,
                order,
                run,
                trial,
                trs_len,
                objective_string,
                count,
                time,
                c,
                trs_string,
                meta_string
            )
        }
    }
}

fn update_data_mcmc<'a, 'b>(
    chain: &mut ParallelTempering<MetaProgramHypothesis<'a, 'b>>,
    top_n: &mut TopN<Box<MetaProgramHypothesisWrapper<'a, 'b>>>,
    mode: LearningMode,
    prior_temp: f64,
    data: &'b [&'b TRSDatum],
    prune_n: usize,
) {
    // 0. Update the top_n.
    for mut h in std::mem::replace(top_n, TopN::new(prune_n)).to_vec() {
        h.data.0.ctl.data = data;
        h.data.0.compute_posterior(data, None);
        h.score = -h.data.0.at_temperature(Temperature::new(prior_temp, 1.0));
        top_n.add(h);
    }

    // 1. Update the chain.
    if data
        .iter()
        .filter(|x| matches!(x, TRSDatum::Full(_)))
        .count()
        > 0
    {
        let best = top_n
            .iter()
            .filter(|x| x.data.0.ctl.mode == mode)
            .min_by(|a, b| {
                a.score
                    .partial_cmp(&b.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        match best {
            Some(best) => {
                chain.set_data(data, true);
                for (_, thread) in chain.pool.iter_mut() {
                    thread.current_mut().clone_from(&best.data.0);
                }
            }
            None => {
                chain.set_data(data, true);
                for (_, thread) in chain.pool.iter_mut() {
                    thread.current_mut().ctl.data = data;
                }
            }
        }
    }
}
