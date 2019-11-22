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
use programinduction::trs::{
    parse_rule, parse_rulecontext, parse_term, parse_trs, Lexicon, ModelParams, TRS,
};
use rand::{seq::SliceRandom, thread_rng, Rng};
use std::{cmp::Ordering, f64, fs::File, io::BufReader, path::PathBuf, process::exit, str};
use term_rewriting::{trace::Trace, Operator, Rule, TRS as UntypedTRS};

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
        .map(|e| e.to_rule(&lex, c))
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

fn test_trace(h_star: &TRS, datum: &Rule, params: &Params) {
    let h_star_trs = h_star.utrs();
    let sig = h_star.lexicon().signature();
    println!("datum: {}", datum.pretty(&sig));
    let mut trace = Trace::new(
        &h_star_trs,
        &sig,
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
            node.term().pretty(&sig),
            node.state(),
            node.log_p(),
            node.children().len(),
        );
    }
}

fn test_enumerator(lex: &mut Lexicon, print: bool) {
    let sig = lex.signature();
    for i in 0..=8 {
        let schema = polytype::TypeSchema::Monotype(lex.fresh_type_variable());
        let enum_terms = lex.enumerate_n_terms(&schema, true, false, i);
        println!("{} terms of length {}", enum_terms.len(), i);
        if print && enum_terms.len() <= 200 {
            for term in &enum_terms {
                println!("    {}", term.pretty(&sig));
            }
        }
    }
    for i in 0..=8 {
        let schema = polytype::TypeSchema::Monotype(lex.fresh_type_variable());
        let enum_rules = lex.enumerate_n_rules(&schema, true, false, i);
        println!("{} rules of length {}", enum_rules.len(), i);
        if print && enum_rules.len() <= 300 {
            for rule in &enum_rules {
                println!("    {}", rule.pretty(&sig));
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
    let sig = lex.signature();
    for term in &terms {
        let schema = polytype::TypeSchema::Monotype(lex.fresh_type_variable());
        let prior = lex
            .logprior_term(term, &schema, params.genetic.atom_weights, false)
            .unwrap();
        println!("{:?},{:.4e}", term.pretty(&sig), prior);
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
    let sig = lex.signature();
    for datum in data {
        let datum_string = datum.pretty(&sig);
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
    let sig = lex.signature();
    for rule in rules {
        let trs = TRS::new(lex, vec![rule.clone()]).unwrap();
        if !UntypedTRS::alphas(&trs.utrs(), &h_star.utrs()) {
            trss.push((trs.clone(), rule.pretty(&sig)));
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
    let sig = lex.signature();
    while rules.len() < n {
        if let Ok(rule) = lex
            .sample_rule_from_context(context.clone(), (1.0, 1.0, 1.0, 1.0), true, 10)
            .drop()
        {
            if !rules.iter().any(|r| Rule::alpha(r, &rule).is_some()) {
                // we don't have the rule yet, so let's see what our likelihood is
                let trs = TRS::new(lex, vec![rule.clone()]).unwrap();
                if trs.log_likelihood(sanity_data, params.model).is_finite() {
                    println!("rule {}: {}", rules.len(), rule.pretty(&sig));
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
    let sig = lex.signature();
    let results = trss
        .into_iter()
        .map(|(trs, label)| {
            let string = trs.utrs().pretty(&sig).replace("\n", "");
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
