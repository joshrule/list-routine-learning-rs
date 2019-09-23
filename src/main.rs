//! [Rust][1] simulations using input/output examples to learn [typed][2] first-order [term rewriting systems][3] that perform list routines.
//!
//! [1]: https://www.rust-lang.org
//! "The Rust Programming Language"
//! [2]: https://en.wikipedia.org/wiki/Hindley%E2%80%93Milner_type_system
//! "Wikipedia - Hindley-Milner Type System"
//! [3]: https://en.wikipedia.org/wiki/Rewriting#Term_rewriting_systems
//! "Wikipedia - Term Rewriting Systems"
extern crate docopt;
extern crate itertools;
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
use itertools::Itertools;
use polytype::Context as TypeContext;
use programinduction::trs::{
    parse_context,
    parse_lexicon,
    parse_rule,
    parse_rulecontext,
    // parse_templates,
    parse_term,
    parse_trs,
    task_by_rewrite,
    // GeneticParams,
    Lexicon,
    ModelParams,
    TRS,
};
use programinduction::GP;
use rand::{
    distributions::{Bernoulli, Distribution},
    seq::SliceRandom,
    thread_rng, Rng,
};
use std::cmp::Ordering;
use std::convert::TryInto;
use std::f64;
use std::fs::{read_to_string, File};
// use std::io;
use std::io::BufReader;
use std::path::PathBuf;
use std::process::exit;
use std::str;
use term_rewriting::{trace::Trace, Operator, Rule, Signature, Term, TRS as UntypedTRS};
use utils::*;

fn main() {
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
    let mut data: Vec<_> = examples
        .iter()
        .map(|e| e.to_rule(&lex, c.clone()))
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

    if true {
        start_section("Testing Trace");
        for (i, datum) in data.iter().enumerate() {
            println!("DATUM {}", i);
            test_trace(&h_star, datum, &params);
        }

        start_section("Testing Probabilistic Model");
        test_logprior(&mut lex, &params);
        println!("");
        test_loglikelihood(&mut lex, &params);
        println!("");
        test_h_star(&h_star, &params, &data[0..1]);

        if false {
            start_section("Testing Enumerator");
            let print_rules = false;
            test_enumerator(&mut lex, print_rules);

            test_computational_model(&mut lex, &mut data[..], h_star.clone(), &params, rng);
        }
    }

    start_section("Initial Population");
    let mut pop = exit_err(
        initialize_population(&lex, &params, rng, &data[0].lhs),
        "couldn't initialize population",
    );
    for (i, (trs, score)) in pop.iter().enumerate() {
        println!("{}: {:.4} {:?}", i, score, trs.to_string());
    }

    start_section("Initial Prediction");
    let prediction = make_prediction(&pop, &data[0].lhs, &params);
    println!(
        "*** prediction: {} -> {} ({})",
        data[0].lhs.pretty(),
        prediction.pretty(),
        data[0].rhs[0].pretty()
    );

    if true {
        start_section("Evolving");
        exit_err(
            evolve(
                &data[..params.simulation.n_examples],
                &mut pop,
                &h_star,
                &lex,
                &params,
                rng,
            ),
            "evolution failed",
        );
    }
}

fn start_section(s: &str) {
    println!("\n{}\n{}", s, "-".repeat(s.len()));
}

fn exit_err<T>(x: Result<T, String>, msg: &str) -> T {
    x.unwrap_or_else(|err| {
        eprintln!("{}: {}", msg, err);
        exit(1);
    })
}

fn str_err<T, U: ToString>(x: Result<T, U>) -> Result<T, String> {
    x.or_else(|err| Err(err.to_string()))
}

fn load_args() -> Result<Params, String> {
    let args: Args = Docopt::new("Usage: sim <args-file>")
        .and_then(|d| d.deserialize())
        .unwrap_or_else(|e| e.exit());
    let args_file = PathBuf::from(args.arg_args_file);
    let toml_string = str_err(read_to_string(args_file))?;
    str_err(toml::from_str(&toml_string))
}

fn path_to_string(dir: &str, file: &str) -> Result<String, String> {
    let path: PathBuf = [dir, file].iter().collect();
    str_err(read_to_string(path))
}

fn load_lexicon(problem_dir: &str, deterministic: bool) -> Result<Lexicon, String> {
    str_err(parse_lexicon(
        &path_to_string(problem_dir, "signature")?,
        &path_to_string(problem_dir, "background")?,
        &path_to_string(problem_dir, "templates")?,
        deterministic,
        TypeContext::default(),
    ))
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
        lex.has_op(Some("C"), 1)
            .or_else(|_| Err(String::from("No target concept"))),
    )
}

fn load_h_star(problem_dir: &str, lex: &mut Lexicon) -> Result<TRS, String> {
    str_err(parse_trs(&path_to_string(problem_dir, "evaluate")?, lex))
}

fn test_trace(h_star: &TRS, datum: &Rule, params: &Params) {
    let h_star_trs = h_star.utrs();
    println!("datum: {}", datum.pretty());
    let mut trace = Trace::new(
        &h_star_trs,
        &datum.lhs,
        params.model.p_observe,
        params.model.max_size,
        params.model.max_depth,
        params.model.strategy,
    );
    trace.rewrite(params.model.max_size.unwrap_or(10));
    for (i, node) in trace.root().iter().enumerate() {
        println!(
            "{}: {}, ({}, {:.5}, {})",
            i,
            node.term().pretty(),
            node.state(),
            node.log_p(),
            node.children().len(),
        );
    }
}

fn test_enumerator(lex: &mut Lexicon, print: bool) {
    for i in 0..=8 {
        let schema = polytype::TypeSchema::Monotype(lex.fresh_type_variable());
        let enum_terms = lex.enumerate_n_terms(&schema, true, false, i);
        println!("{} terms of length {}", enum_terms.len(), i);
        if print && enum_terms.len() <= 200 {
            for term in &enum_terms {
                println!("    {}", term.pretty());
            }
        }
    }
    for i in 0..=8 {
        let schema = polytype::TypeSchema::Monotype(lex.fresh_type_variable());
        let enum_rules = lex.enumerate_n_rules(&schema, true, false, i);
        println!("{} rules of length {}", enum_rules.len(), i);
        if print && enum_rules.len() <= 300 {
            for rule in &enum_rules {
                println!("    {}", rule.pretty());
            }
        }
    }
}

fn test_h_star(h_star: &TRS, params: &Params, data: &[Rule]) {
    println!("trs: {}", h_star.to_string());
    let prior = h_star.log_prior(params.model);
    let ll = h_star.log_likelihood(data, params.model);
    println!("prior: {}", prior);
    println!("likelihood: {}", ll);
    println!("posterior: {}", prior + ll);
}

fn test_logprior(lex: &mut Lexicon, params: &Params) {
    let trss = vec![
        str_err(parse_trs("C(x_) = NIL;", lex)).unwrap(),
        str_err(parse_trs("C(x_) = SINGLETON(DIGIT(3));", lex)).unwrap(),
        str_err(parse_trs("C(x_) = SINGLETON(DECC(DIGIT(3) 7));", lex)).unwrap(),
        str_err(parse_trs("C(x_) = CONS(DECC(DIGIT(1) 9) CONS(DECC(DIGIT(1) 2) CONS(DECC(DIGIT(3) 3) CONS(DECC(DIGIT(2) 4) CONS(DIGIT(5) NIL)))));", lex)).unwrap(),
    ];
    println!("trs,prior");
    for trs in &trss {
        let prior = trs.log_prior(params.model);
        println!("{:?},{:.4e}", trs.to_string(), prior);
    }

    let terms = vec![
        str_err(parse_term("NIL", lex)).unwrap(),
        str_err(parse_term("SINGLETON(DIGIT(3))", lex)).unwrap(),
        str_err(parse_term("SINGLETON(DECC(DIGIT(3) 7))", lex)).unwrap(),
        str_err(parse_term("CONS(DIGIT(3) NIL)", lex)).unwrap(),
        str_err(parse_term("CONS(DECC(DIGIT(3) 7) NIL)", lex)).unwrap(),
        str_err(parse_term("CONS(DECC(DIGIT(1) 9) CONS(DECC(DIGIT(1) 2) CONS(DECC(DIGIT(3) 3) CONS(DECC(DIGIT(2) 4) CONS(DIGIT(5) NIL)))))", lex)).unwrap(),
    ];
    println!("\nterm,prior");
    for term in &terms {
        let schema = polytype::TypeSchema::Monotype(lex.fresh_type_variable());
        let prior = lex
            .logprior_term(term, &schema, params.genetic.atom_weights, false)
            .unwrap();
        println!("{:?},{:.4e}", term.pretty(), prior);
    }
}

fn test_loglikelihood(lex: &mut Lexicon, params: &Params) {
    let trs = str_err(parse_trs("C(x_) = CONS(DECC(DIGIT(1) 9) CONS(DECC(DIGIT(1) 2) CONS(DECC(DIGIT(3) 3) CONS(DECC(DIGIT(2) 4) CONS(DIGIT(5) NIL)))));", lex)).unwrap();
    let data = vec![
        str_err(parse_rule("C(CONS(DIGIT(5) CONS(DIGIT(3) NIL))) = NIL", lex)).unwrap(),
        str_err(parse_rule("C(CONS(DIGIT(5) CONS(DIGIT(3) NIL))) = CONS(DIGIT(7) NIL)", lex)).unwrap(),
        str_err(parse_rule("C(CONS(DIGIT(5) CONS(DIGIT(3) NIL))) = CONS(DECC(DIGIT(1) 1) NIL)", lex)).unwrap(),
        str_err(parse_rule("C(CONS(DIGIT(5) CONS(DIGIT(3) NIL))) = CONS(DECC(DIGIT(1) 1) CONS(DIGIT(3) NIL))", lex)).unwrap(),
        str_err(parse_rule("C(CONS(DIGIT(5) CONS(DIGIT(3) NIL))) = CONS(DECC(DIGIT(1) 1) CONS(DIGIT(3) CONS(DIGIT(7) CONS(DECC(DIGIT(4) 4) CONS(DECC(DIGIT(2) 7) CONS(DIGIT(3) NIL))))))", lex)).unwrap(),
        str_err(parse_rule("C(CONS(DIGIT(5) CONS(DIGIT(3) NIL))) = CONS(DECC(DIGIT(1) 1) CONS(DIGIT(3) CONS(DIGIT(7) NIL)))", lex)).unwrap(),
        str_err(parse_rule("C(CONS(DIGIT(5) CONS(DIGIT(3) NIL))) = CONS(DECC(DIGIT(1) 1) CONS(DIGIT(3) CONS(DIGIT(7) CONS(DECC(DIGIT(4) 4) CONS(DECC(DIGIT(2) 7) NIL)))))", lex)).unwrap(),
        str_err(parse_rule("C(CONS(DIGIT(5) CONS(DIGIT(3) NIL))) = CONS(DECC(DIGIT(1) 9) NIL)", lex)).unwrap(),
        str_err(parse_rule("C(CONS(DIGIT(5) CONS(DIGIT(3) NIL))) = CONS(DECC(DIGIT(1) 9) CONS(DIGIT(3) NIL))", lex)).unwrap(),
        str_err(parse_rule("C(CONS(DIGIT(5) CONS(DIGIT(3) NIL))) = CONS(DECC(DIGIT(1) 1) CONS(DECC(DIGIT(1) 2) CONS(DIGIT(7) NIL)))", lex)).unwrap(),
        str_err(parse_rule("C(CONS(DIGIT(5) CONS(DIGIT(3) NIL))) = CONS(DECC(DIGIT(1) 9) CONS(DIGIT(3) CONS(DIGIT(7) NIL)))", lex)).unwrap(),
        str_err(parse_rule("C(CONS(DIGIT(5) CONS(DIGIT(3) NIL))) = CONS(DECC(DIGIT(1) 2) CONS(DECC(DIGIT(3) 3) CONS(DIGIT(5) CONS(DECC(DIGIT(1) 9) CONS(DECC(DIGIT(2) 4) NIL)))))", lex)).unwrap(),
        str_err(parse_rule("C(CONS(DIGIT(5) CONS(DIGIT(3) NIL))) = CONS(DECC(DIGIT(1) 9) CONS(DECC(DIGIT(1) 2) NIL))", lex)).unwrap(),
        str_err(parse_rule("C(CONS(DIGIT(5) CONS(DIGIT(3) NIL))) = CONS(DECC(DIGIT(1) 1) CONS(DECC(DIGIT(1) 2) CONS(DECC(DIGIT(3) 3) NIL)))", lex)).unwrap(),
        str_err(parse_rule("C(CONS(DIGIT(5) CONS(DIGIT(3) NIL))) = CONS(DECC(DIGIT(1) 9) CONS(DIGIT(3) CONS(DIGIT(5) NIL)))", lex)).unwrap(),
        str_err(parse_rule("C(CONS(DIGIT(5) CONS(DIGIT(3) NIL))) = CONS(DECC(DIGIT(1) 9) CONS(DECC(DIGIT(3) 3) CONS(DIGIT(5) NIL)))", lex)).unwrap(),
        str_err(parse_rule("C(CONS(DIGIT(5) CONS(DIGIT(3) NIL))) = CONS(DECC(DIGIT(1) 9) CONS(DECC(DIGIT(1) 2) CONS(DECC(DIGIT(3) 3) CONS(DECC(DIGIT(2) 4) CONS(DIGIT(5) NIL)))))", lex)).unwrap(),
    ];
    println!("trs: {:?}", trs.to_string());
    println!("datum,likelihood");
    for datum in data {
        let datum_string = datum.pretty();
        let likelihood = trs.log_likelihood(&[datum], params.model);
        println!("{:?},{:.4e}", datum_string, likelihood);
    }
}

fn test_computational_model<R: Rng>(
    lex: &mut Lexicon,
    data: &mut [Rule],
    h_star: TRS,
    params: &Params,
    rng: &mut R,
) {
    // create set of comparison TRSs: empty TRS and h*
    let mut trss = vec![
        (parse_trs("", lex).unwrap(), "empty".to_string()),
        (h_star.clone(), "correct".to_string()),
    ];

    // create trss from sampled rules
    let n = 100;
    let rules = sample_n_rules(n, lex, &params, &data[..10]);
    list_rule_lengths(&rules);
    for rule in rules {
        let trs = TRS::new(lex, vec![rule.clone()]).unwrap();
        if !UntypedTRS::alphas(&trs.utrs(), &h_star.utrs()) {
            trss.push((trs.clone(), rule.pretty()));
        }
    }

    let now = std::time::SystemTime::now();
    // score trss
    for idx in 0..100 {
        data.shuffle(rng);
        compare_posteriors_loop(
            idx,
            "NA".to_string(),
            lex,
            trss.clone(),
            &data[..5],
            params.model,
        );
    }
    match now.elapsed() {
        Ok(elapsed) => println!("{}s", (elapsed.as_nanos() as f64) / (1e9 as f64)),
        Err(_) => println!("timekeeping failed"),
    }
}

fn sample_n_rules(n: usize, lex: &mut Lexicon, params: &Params, sanity_data: &[Rule]) -> Vec<Rule> {
    let mut rules = Vec::with_capacity(n);
    let context = parse_rulecontext("C(CONS([!] [!])) = [!]", lex).unwrap();
    while rules.len() < n {
        if let Ok(rule) =
            lex.sample_rule_from_context(context.clone(), (1.0, 1.0, 1.0, 1.0), true, 10)
        {
            if !rules.iter().any(|r| Rule::alpha(r, &rule).is_some()) {
                // we don't have the rule yet, so let's see what our likelihood is
                let trs = TRS::new(lex, vec![rule.clone()]).unwrap();
                if trs.log_likelihood(sanity_data, params.model).is_finite() {
                    println!("rule {}: {}", rules.len(), rule.pretty());
                    rules.push(rule);
                }
            }
        }
    }
    rules
}

fn list_rule_lengths(rules: &[Rule]) {
    println!("rule lengths");
    for (length, n) in &rules
        .iter()
        .map(|rule| rule.size())
        .sorted()
        .group_by(|x| *x)
    {
        println!("- {} rules of length {}", n.count(), length);
    }
}

fn compare_posteriors_loop(
    idx: usize,
    rule: String,
    lex: &mut Lexicon,
    trss: Vec<(TRS, String)>,
    data: &[Rule],
    params: ModelParams,
) {
    for n_data in 1..=data.len() {
        compare_posteriors(idx, &rule, lex, trss.clone(), &data[..n_data], params)
    }
}

fn compare_posteriors(
    idx: usize,
    rule: &str,
    lex: &mut Lexicon,
    mut trss: Vec<(TRS, String)>,
    data: &[Rule],
    params: ModelParams,
) {
    // create your alternative hypotheses
    let memorized = TRS::new(lex, data.to_vec()).unwrap();
    trss.push((memorized, "memorized".to_string()));
    // score each hypothesis
    let results = trss
        .into_iter()
        .map(|(trs, label)| {
            let string = trs.utrs().pretty().replace("\n", "");
            let prior = trs.log_prior(params);
            let ll = trs.log_likelihood(data, params);
            let posterior = params.p_temp * prior + params.l_temp * ll;
            (string, prior, ll, posterior, label)
        })
        .sorted_by(|x, y| x.3.partial_cmp(&y.3).unwrap_or(Ordering::Equal))
        .collect_vec();
    // renormalize the set and print results
    let z = logsumexp(&results.iter().map(|x| x.3).collect_vec());
    for (string, prior, ll, posterior, label) in results {
        let norm_post = posterior - z;
        println!(
            "\"{}\",{},{},{},{},{},{},\"{}\",\"{}\"",
            rule,
            idx,
            data.len(),
            prior,
            ll,
            posterior,
            norm_post,
            label,
            string
        );
    }
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
    let schema = lex.infer_term(input).unwrap();
    let mut rules = Vec::with_capacity(params.gp.population_size);
    while rules.len() < rules.capacity() {
        let mut pop = lex.genesis(&params.genetic, rng, 1, &schema);
        let trs = pop[0].utrs();
        let unique = !rules
            .iter()
            .any(|(ptrs, _): &(TRS, _)| UntypedTRS::alphas(&trs, &ptrs.utrs()));
        if unique {
            let mut trace = Trace::new(
                &trs,
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
                    let mut sig = Signature::default();
                    if UntypedTRS::convert_list_to_string(&x.term(), &mut sig).is_some() {
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
    Ok(rules)
}

fn make_prediction(pop: &[(TRS, f64)], input: &Term, params: &Params) -> Term {
    let best_trs = pop
        .iter()
        .min_by(|(_, x), (_, y)| x.partial_cmp(y).or_else(|| Some(Ordering::Less)).unwrap())
        .unwrap()
        .0
        .utrs();
    let mut trace = Trace::new(
        &best_trs,
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
    pop: &mut Vec<(TRS, f64)>,
    h_star: &TRS,
    lex: &Lexicon,
    params: &Params,
    rng: &mut R,
) -> Result<(), String> {
    println!("n_data,generation,rank,nlposterior,h_star_nlposterior,trs");
    for n_data in 1..=(data.len() - 1) {
        let examples = {
            let mut exs = vec![];
            for i in 0..n_data {
                let idx = n_data - i - 1;
                let weight = params.simulation.decay.powi(idx.try_into().unwrap());
                let dist = Bernoulli::new(weight).unwrap();
                if dist.sample(rng) {
                    exs.push(data[i].clone());
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
        println!(
            "*** prediction: {} -> {} ({})",
            input.pretty(),
            prediction.pretty(),
            output.pretty()
        );
    }
    Ok(())
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
                0..=9 => Value::make_digit(lex, num),
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
