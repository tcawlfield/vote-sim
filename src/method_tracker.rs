use std::sync::Arc;

use arrow_array::builder::PrimitiveBuilder;
use arrow_array::types::{Float64Type, Int32Type};
use arrow_array::{ArrayRef, Float64Array, Int32Array, StructArray};
use arrow_schema::{DataType, Field, Fields};
use meansd::MeanSD;

use crate::methods::{Method, MethodSim, WinnerAndRunnerup};
use crate::sim::Sim;

pub struct MethodTracker {
    pub method: Box<dyn MethodSim>,
    ntrials: usize,
    ntrials_subopt: usize,
    mean_regret: MeanSD,
    mean_subopt_regret: MeanSD,
    result_bldr: PrimitiveBuilder<Float64Type>,
    winner_bldr: PrimitiveBuilder<Int32Type>,
}

impl MethodTracker {
    pub fn new(method: &Method, sim: &Sim, max_trials: usize) -> MethodTracker {
        MethodTracker {
            method: method.new_sim(sim),
            ntrials: 0,
            ntrials_subopt: 0,
            mean_regret: MeanSD::default(),
            mean_subopt_regret: MeanSD::default(),
            result_bldr: Float64Array::builder(max_trials),
            winner_bldr: Int32Array::builder(max_trials),
        }
    }

    pub fn elect(
        &mut self,
        sim: &Sim,
        honest_rslt: Option<WinnerAndRunnerup>,
    ) -> WinnerAndRunnerup {
        let mut result = self.method.elect(sim, honest_rslt);

        if result.is_tied() {
            result = sim.break_tie_with_plurality(&result);
        }

        self.ntrials += 1;
        let regret = sim.regrets[result.winner.cand];
        self.mean_regret.update(regret);
        if regret > 0.0 {
            self.ntrials_subopt += 1;
            self.mean_subopt_regret.update(regret);
        }

        self.result_bldr.append_value(regret);
        self.winner_bldr
            .append_value(sim.regret_rank[result.winner.cand] as i32);
        result
    }

    pub fn colname(&self) -> String {
        self.method.colname()
    }

    pub fn data_type() -> DataType {
        DataType::Struct(Fields::from(vec![
            Arc::new(Field::new("winner", DataType::Int32, false)),
            Arc::new(Field::new("regret", DataType::Float64, false)),
        ]))
    }

    // pub fn get_field(&self) -> Field {
    //     Field::new(self.method.colname(), Self::data_type(), false)
    // }

    pub fn get_column(&mut self) -> arrow_array::ArrayRef {
        // Arc::new(self.result_bldr.finish())
        let struct_array = StructArray::from(vec![
            (
                Arc::new(Field::new("winner", DataType::Int32, false)),
                Arc::new(self.winner_bldr.finish()) as ArrayRef,
            ),
            (
                Arc::new(Field::new("regret", DataType::Float64, false)),
                Arc::new(self.result_bldr.finish()) as ArrayRef,
            ),
        ]);
        Arc::new(struct_array)
    }

    #[allow(dead_code)]
    pub fn report(&self) {
        let frac_suboptimal = self.ntrials_subopt as f64 / self.ntrials as f64;
        println!(
            "Method {}: Avg Regret: {}, σ: {}, Frac suboptimal winner: {}, avg subopt regret: {}",
            self.method.name(),
            self.mean_regret.mean(),
            self.mean_regret.sstdev(),
            frac_suboptimal,
            self.mean_subopt_regret.mean(),
        )
    }

    pub fn sendable_report(&self) -> SendableMethodReport {
        SendableMethodReport {
            name: self.method.name(),
            ntrials: self.ntrials,
            ntrials_subopt: self.ntrials_subopt,
            mean_regret: self.mean_regret.clone(),
            mean_subopt_regret: self.mean_subopt_regret.clone(),
        }
    }
}

pub struct SendableMethodReport {
    name: String,
    ntrials: usize,
    ntrials_subopt: usize,
    mean_regret: MeanSD,
    mean_subopt_regret: MeanSD,
}

impl SendableMethodReport {
    pub fn combine(&mut self, other: &Self) {
        assert!(self.name == other.name);
        self.ntrials += other.ntrials;
        self.ntrials_subopt += other.ntrials_subopt;
        self.mean_regret += other.mean_regret;
        self.mean_subopt_regret += other.mean_subopt_regret;
    }

    pub fn report(&self) {
        let frac_suboptimal = self.ntrials_subopt as f64 / self.ntrials as f64;
        println!(
            "Method {}: Avg Regret: {}, σ: {}, Frac suboptimal winner: {}, avg subopt regret: {}, {} elections",
            self.name,
            self.mean_regret.mean(),
            self.mean_regret.sstdev(),
            frac_suboptimal,
            self.mean_subopt_regret.mean(),
            self.ntrials,
        );
    }
}
