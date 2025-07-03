use crate::sim::Sim;
use ndarray::Array2;
use rand::rngs::ThreadRng;
use std::fmt;

mod irrational;
mod issues;
mod likability;

pub use irrational::Irrational;
pub use issues::{new_issues_sim, Issue};
pub use likability::Likability;

pub trait ConsiderationSim: fmt::Debug {
    fn add_to_scores(&mut self, scores: &mut Array2<f64>, rng: &mut ThreadRng, verbose: bool);
    fn get_dim(&self) -> usize;
    fn get_name(&self) -> String;
    fn push_posn_elements(&self, report: &mut dyn FnMut(f64, bool), final_candidates: &Vec<usize>);
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub enum Consideration {
    Likability(Likability),
    Issues(Vec<Issue>),
    Irrational(Irrational),
}

impl Consideration {
    pub fn new_sim(&self, sim: &Sim) -> Box<dyn ConsiderationSim> {
        match self {
            Consideration::Likability(c) => Box::new(c.new_sim(sim)),
            Consideration::Issues(issues) => Box::new(new_issues_sim(issues.clone(), sim)),
            Consideration::Irrational(c) => Box::new(c.new_sim(sim)),
        }
    }
}
