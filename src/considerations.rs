use std::fmt;
use ndarray::Array2;
use rand::rngs::ThreadRng;
use crate::methods::ElectResult;
use crate::sim::Sim;

mod likability;
mod issues;

pub use likability::Likability;
pub use issues::Issues;

pub trait ConsiderationSim: fmt::Debug {
    fn add_to_scores(&mut self, scores: &mut Array2<f64>, rng: &mut ThreadRng, verbose: bool);
    fn get_dim(&self) -> usize;
    fn get_name(&self) -> String;
    fn push_posn_elements(
        &self,
        report: &mut dyn FnMut(f64, bool),
        final_candidates: Option<&Vec<ElectResult>>,
    );
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub enum Consideration {
    Likability(Likability),
    Issues(Issues),
}

impl Consideration {
    pub fn as_sim(&self, sim: &Sim) -> Box<dyn ConsiderationSim> {
        match self {
            Consideration::Likability(c) => Box::new(c.new_sim(sim)),
            Consideration::Issues(c) => Box::new(c.new_sim(sim)),
        }
    }
}
