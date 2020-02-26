//! [Rust][1] simulations using input/output examples to learn [typed][2] first-order [term rewriting systems][3] that perform list routines.
//!
//! [1]: https://www.rust-lang.org
//! "The Rust Programming Language"
//! [2]: https://en.wikipedia.org/wiki/Hindley%E2%80%93Milner_type_system
//! "Wikipedia - Hindley-Milner Type System"
//! [3]: https://en.wikipedia.org/wiki/Rewriting#Term_rewriting_systems
//! "Wikipedia - Term Rewriting Systems"

use itertools::Itertools;
use list_routine_learning_rs::*;
use programinduction::trs::Lexicon;
use rand::{
    distributions::{Distribution, Uniform},
    thread_rng, Rng,
};
use std::f64;
use term_rewriting::{PStringDist, PStringIncorrect, TRS as UntypedTRS};

fn main() {
    // Set random number generator.
    let rng = &mut thread_rng();
    // create a likelihood.
    let t_max = 10;
    let d_max = 10;
    let dist = PStringDist {
        // Probability of a single insertion.
        beta: 0.01,
        // Probability of inserting a given symbol.
        p_insertion: 0.01 / 100.0,
        // Probability of deleting an existing element.
        p_deletion: 0.005,
        // Probability of correct substitution for an existing element.
        p_correct_sub: 0.99,
        // Distribution over incorrect substitutions for an existing element.
        p_incorrect_sub: PStringIncorrect::Constant(0.005 / 99.0),
    };
    //
    let lex = exit_err(load_lexicon("."), "Failed to load lexicon");
    println!("{}\n", lex);
    // test the likelihood function
    let xs = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
    let mut costs = vec![];
    for i in 0..=10 {
        let cost = blah(&Value::IntList(xs[..i].to_vec()), dist, t_max, d_max, &lex);
        println!(
            "correct cost for length {} is {:.4} ({:.4})",
            i,
            cost,
            cost.exp()
        );
        costs.push(cost);
    }
    let mean_cost = costs.iter().sum::<f64>() / (costs.len() as f64);
    println!(
        "mean correct cost is {:.4} ({:.4})",
        mean_cost,
        mean_cost.exp()
    );
    let n_meta = 10;
    let n_lists = 10000;
    // correct pairings
    let meta_costs = (0..n_meta)
        .map(|_| collect_costs(n_lists, dist, t_max, d_max, &lex, rng, true))
        .collect_vec();
    let mean_cost = meta_costs.iter().sum::<f64>() / (n_meta as f64);
    println!(
        "mean correct cost over {} samples: {:.6e}",
        meta_costs.len(),
        mean_cost,
    );
    // incorrect pairings
    let meta_costs = (0..n_meta)
        .map(|_| collect_costs(n_lists, dist, t_max, d_max, &lex, rng, false))
        .collect_vec();
    let mean_cost = meta_costs.iter().sum::<f64>() / (n_meta as f64);
    println!(
        "mean incorrect cost over {} samples: {:.6e}",
        meta_costs.len(),
        mean_cost,
    );
}

fn collect_costs<R: Rng>(
    n_lists: usize,
    dist: PStringDist,
    t_max: usize,
    d_max: usize,
    lex: &Lexicon,
    rng: &mut R,
    same: bool,
) -> f64 {
    let mut costs = vec![];
    while costs.len() < n_lists {
        let m = sample_list(rng);
        let n = if same { m.clone() } else { sample_list(rng) };
        if same || m != n {
            let cost1 = blah2(
                &Value::IntList(m.clone()),
                &Value::IntList(n.clone()),
                dist,
                t_max,
                d_max,
                &lex,
            );
            costs.push(cost1);
            if !same {
                let cost2 = blah2(
                    &Value::IntList(n),
                    &Value::IntList(m),
                    dist,
                    t_max,
                    d_max,
                    &lex,
                );
                costs.push(cost2);
            }
        }
    }
    let mean = (costs.iter().sum::<f64>()) / (costs.len() as f64);
    println!("mean cost over {} pairings: {:.4e}", costs.len(), mean);
    mean
}

fn sample_list<R: Rng>(rng: &mut R) -> Vec<usize> {
    let length_dist = Uniform::from(0..=10);
    let length = length_dist.sample(rng);
    let element_dist = Uniform::from(0..=99);
    let mut xs = vec![];
    for _ in 0..=length {
        xs.push(element_dist.sample(rng));
    }
    xs
}

fn blah(value: &Value, dist: PStringDist, t_max: usize, d_max: usize, lex: &Lexicon) -> f64 {
    let list = value.to_term(&lex, None).unwrap();
    let sig = lex.signature();
    UntypedTRS::p_list(&list, &list, dist, t_max, d_max, &sig)
}

fn blah2(
    v1: &Value,
    v2: &Value,
    dist: PStringDist,
    t_max: usize,
    d_max: usize,
    lex: &Lexicon,
) -> f64 {
    let l1 = v1.to_term(&lex, None).unwrap();
    let l2 = v2.to_term(&lex, None).unwrap();
    let sig = lex.signature();
    UntypedTRS::p_list(&l1, &l2, dist, t_max, d_max, &sig)
}

// fn test_logprior(lex: &mut Lexicon, params: &Params) {
//     let trss = vec![
//         str_err(parse_trs("C(x_) = NIL;", lex)).unwrap(),
//         str_err(parse_trs("C(x_) = SINGLETON(DIGIT(3));", lex)).unwrap(),
//         str_err(parse_trs("C(x_) = SINGLETON(DECC(DIGIT(3) 7));", lex)).unwrap(),
//         str_err(parse_trs("C(x_) = CONS(DECC(DIGIT(1) 9) CONS(DECC(DIGIT(1) 2) CONS(DECC(DIGIT(3) 3) CONS(DECC(DIGIT(2) 4) CONS(DIGIT(5) NIL)))));", lex)).unwrap(),
//     ];
//     println!("trs,prior");
//     for trs in &trss {
//         let prior = trs.log_prior(params.model);
//         println!("{:?},{:.4e}", trs.to_string(), prior);
//     }
//
//     let terms = vec![
//         str_err(parse_term("NIL", lex)).unwrap(),
//         str_err(parse_term("SINGLETON(DIGIT(3))", lex)).unwrap(),
//         str_err(parse_term("SINGLETON(DECC(DIGIT(3) 7))", lex)).unwrap(),
//         str_err(parse_term("CONS(DIGIT(3) NIL)", lex)).unwrap(),
//         str_err(parse_term("CONS(DECC(DIGIT(3) 7) NIL)", lex)).unwrap(),
//         str_err(parse_term("CONS(DECC(DIGIT(1) 9) CONS(DECC(DIGIT(1) 2) CONS(DECC(DIGIT(3) 3) CONS(DECC(DIGIT(2) 4) CONS(DIGIT(5) NIL)))))", lex)).unwrap(),
//     ];
//     println!("\nterm,prior");
//     let sig = lex.signature();
//     for term in &terms {
//         let schema = polytype::TypeSchema::Monotype(lex.fresh_type_variable());
//         let prior = lex
//             .logprior_term(term, &schema, params.genetic.atom_weights, false)
//             .unwrap();
//         println!("{:?},{:.4e}", term.pretty(&sig), prior);
//     }
// }

// fn test_loglikelihood(lex: &mut Lexicon, params: &Params) {
//     let trs = str_err(parse_trs("C(x_) = CONS(DECC(DIGIT(1) 9) CONS(DECC(DIGIT(1) 2) CONS(DECC(DIGIT(3) 3) CONS(DECC(DIGIT(2) 4) CONS(DIGIT(5) NIL)))));", lex)).unwrap();
//     let data = vec![
//         str_err(parse_rule("C(CONS(DIGIT(5) CONS(DIGIT(3) NIL))) = NIL", lex)).unwrap(),
//         str_err(parse_rule("C(CONS(DIGIT(5) CONS(DIGIT(3) NIL))) = CONS(DIGIT(7) NIL)", lex)).unwrap(),
//         str_err(parse_rule("C(CONS(DIGIT(5) CONS(DIGIT(3) NIL))) = CONS(DECC(DIGIT(1) 1) NIL)", lex)).unwrap(),
//         str_err(parse_rule("C(CONS(DIGIT(5) CONS(DIGIT(3) NIL))) = CONS(DECC(DIGIT(1) 1) CONS(DIGIT(3) NIL))", lex)).unwrap(),
//         str_err(parse_rule("C(CONS(DIGIT(5) CONS(DIGIT(3) NIL))) = CONS(DECC(DIGIT(1) 1) CONS(DIGIT(3) CONS(DIGIT(7) CONS(DECC(DIGIT(4) 4) CONS(DECC(DIGIT(2) 7) CONS(DIGIT(3) NIL))))))", lex)).unwrap(),
//         str_err(parse_rule("C(CONS(DIGIT(5) CONS(DIGIT(3) NIL))) = CONS(DECC(DIGIT(1) 1) CONS(DIGIT(3) CONS(DIGIT(7) NIL)))", lex)).unwrap(),
//         str_err(parse_rule("C(CONS(DIGIT(5) CONS(DIGIT(3) NIL))) = CONS(DECC(DIGIT(1) 1) CONS(DIGIT(3) CONS(DIGIT(7) CONS(DECC(DIGIT(4) 4) CONS(DECC(DIGIT(2) 7) NIL)))))", lex)).unwrap(),
//         str_err(parse_rule("C(CONS(DIGIT(5) CONS(DIGIT(3) NIL))) = CONS(DECC(DIGIT(1) 9) NIL)", lex)).unwrap(),
//         str_err(parse_rule("C(CONS(DIGIT(5) CONS(DIGIT(3) NIL))) = CONS(DECC(DIGIT(1) 9) CONS(DIGIT(3) NIL))", lex)).unwrap(),
//         str_err(parse_rule("C(CONS(DIGIT(5) CONS(DIGIT(3) NIL))) = CONS(DECC(DIGIT(1) 1) CONS(DECC(DIGIT(1) 2) CONS(DIGIT(7) NIL)))", lex)).unwrap(),
//         str_err(parse_rule("C(CONS(DIGIT(5) CONS(DIGIT(3) NIL))) = CONS(DECC(DIGIT(1) 9) CONS(DIGIT(3) CONS(DIGIT(7) NIL)))", lex)).unwrap(),
//         str_err(parse_rule("C(CONS(DIGIT(5) CONS(DIGIT(3) NIL))) = CONS(DECC(DIGIT(1) 2) CONS(DECC(DIGIT(3) 3) CONS(DIGIT(5) CONS(DECC(DIGIT(1) 9) CONS(DECC(DIGIT(2) 4) NIL)))))", lex)).unwrap(),
//         str_err(parse_rule("C(CONS(DIGIT(5) CONS(DIGIT(3) NIL))) = CONS(DECC(DIGIT(1) 9) CONS(DECC(DIGIT(1) 2) NIL))", lex)).unwrap(),
//         str_err(parse_rule("C(CONS(DIGIT(5) CONS(DIGIT(3) NIL))) = CONS(DECC(DIGIT(1) 1) CONS(DECC(DIGIT(1) 2) CONS(DECC(DIGIT(3) 3) NIL)))", lex)).unwrap(),
//         str_err(parse_rule("C(CONS(DIGIT(5) CONS(DIGIT(3) NIL))) = CONS(DECC(DIGIT(1) 9) CONS(DIGIT(3) CONS(DIGIT(5) NIL)))", lex)).unwrap(),
//         str_err(parse_rule("C(CONS(DIGIT(5) CONS(DIGIT(3) NIL))) = CONS(DECC(DIGIT(1) 9) CONS(DECC(DIGIT(3) 3) CONS(DIGIT(5) NIL)))", lex)).unwrap(),
//         str_err(parse_rule("C(CONS(DIGIT(5) CONS(DIGIT(3) NIL))) = CONS(DECC(DIGIT(1) 9) CONS(DECC(DIGIT(1) 2) CONS(DECC(DIGIT(3) 3) CONS(DECC(DIGIT(2) 4) CONS(DIGIT(5) NIL)))))", lex)).unwrap(),
//     ];
//     println!("trs: {:?}", trs.to_string());
//     println!("datum,likelihood");
//     let sig = lex.signature();
//     for datum in data {
//         let datum_string = datum.pretty(&sig);
//         let likelihood = trs.log_likelihood(&[datum], params.model);
//         println!("{:?},{:.4e}", datum_string, likelihood);
//     }
// }

pub fn logsumexp(lps: &[f64]) -> f64 {
    let largest = lps.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let x = lps.iter().map(|lp| (lp - largest).exp()).sum::<f64>().ln();
    largest + x
}
