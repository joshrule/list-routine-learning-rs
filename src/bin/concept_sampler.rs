//! A utility for sampling list routines

use itertools::Itertools;
use list_routine_learning_rs::*;
use programinduction::trs::{parse_rulecontext, parse_term, Lexicon, TRS};
use std::collections::HashMap;
use term_rewriting::{Rule, RuleContext, Term, TRS as UTRS};

fn main() {
    let deterministic = true;
    let dir = "trs/dsl";
    let n = 100;

    start_section("Loading Lexicon");
    let lex = exit_err(load_lexicon(&dir, deterministic), "Failed to load lexicon");
    println!("{}", lex);

    start_section("Sampling Concepts");
    sample_systems(&lex, n);
}

fn sample_systems(lex: &Lexicon, n: usize) -> Vec<TRS> {
    let mut trss = Vec::with_capacity(n);
    while trss.len() < n {
        if let Some(trs) = sample_system(lex) {
            if !trss
                .iter()
                .any(|t: &TRS| UTRS::alphas(&t.utrs(), &trs.utrs()))
            {
                println!("{}.", trss.len());
                for rule in &trs.utrs().rules {
                    println!("  {}", rule.pretty(&lex.signature()));
                }
                trss.push(trs);
            }
        }
    }
    trss
}

fn sample_system(lex: &Lexicon) -> Option<TRS> {
    let snapshot = lex.snapshot();
    let mut symbols = ["F", "G", "H"]
        .iter()
        .map(|x| parse_term(x, lex).unwrap())
        .collect();
    let mut stack = Vec::with_capacity(3);
    let mut rules = Vec::with_capacity(4);

    // create a stack of symbols needing rules
    stack.push(parse_term("C", lex).unwrap());
    println!(
        "stack is now: {}",
        stack.iter().map(|x| x.pretty(&lex.signature())).join(", ")
    );

    // Until the stack is empty, pop from the stack
    while let Some(symbol) = stack.pop() {
        // sample a rule for that symbol
        println!("looking for rule for {}", symbol.pretty(&lex.signature()));
        match sample_rule_from_symbol(lex, &symbol) {
            Some(ref rule) if rule.variables().is_empty() => return None,
            Some(rule) => {
                println!("  sampled: {}", rule.pretty(&lex.signature()));
                // push symbols needing definition onto the stack
                update_stack(&rule, &mut symbols, &mut stack);
                // dump the new rule into the system
                rules.push(rule);
            }
            None => {
                println!("  Failed!");
                return None;
            }
        }
        println!(
            "stack is now: {}",
            stack.iter().map(|x| x.pretty(&lex.signature())).join(", ")
        );
    }

    // create the TRS and cleanup the TypeContext
    let trs = TRS::new(lex, rules);
    lex.rollback(snapshot);
    trs.ok()
}

fn update_stack(rule: &Rule, candidates: &mut Vec<Term>, stack: &mut Vec<Term>) {
    let mut subterms = rule
        .subterms()
        .into_iter()
        .map(|(x, _)| x)
        .filter(|x| candidates.contains(&x))
        .cloned()
        .collect_vec();
    candidates.retain(|candidate| !subterms.contains(candidate));
    stack.append(&mut subterms);
}

fn sample_rule_from_symbol(lex: &Lexicon, subterm: &Term) -> Option<Rule> {
    let invent = false;
    let max_size = 30;
    let atom_weights = (1.0, 2.0, 1.0, 1.0);
    let context = context_from_symbol(lex, subterm)?;
    lex.sample_rule_from_context(context, atom_weights, invent, max_size)
        .keep()
        .ok()
}

fn context_from_symbol(lex: &Lexicon, subterm: &Term) -> Option<RuleContext> {
    let var_names = vec!["x_", "y_", "z_", "w_", "v_", "u_", "t_", "s_", "r_"];
    let schema = lex.infer_term(subterm, &mut HashMap::new()).keep().ok()?;
    let subterm_type = lex.instantiate(&schema);
    let args_len = subterm_type.args().map_or(0, |x| x.len());
    let args = &var_names[0..args_len];
    let context_string = format!(
        "{} {} = [!]",
        subterm.pretty(&lex.signature()),
        args.join(" ")
    );
    parse_rulecontext(&context_string, lex).ok()
}
