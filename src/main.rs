use std::error::Error;
use std::ffi::OsString;

use clap::Parser;
use std::process;

// Local libraries
mod config;
mod consideration;
mod cov_matrix;
mod method_tracker;
mod methods;
mod run;
mod sim;

use consideration::*;
use method_tracker::MethodTracker;
use methods::*;
use sim::Sim;

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Args {
    /// Number of trials
    #[arg(short, long, default_value_t = 1)]
    trials: usize,

    /// Config file
    #[arg(short, long, default_value = OsString::from("configs/default.toml"))]
    config: OsString,

    /// Output file
    #[arg(short, long)]
    outfile: Option<OsString>,

    /// Number of voters (override config)
    #[arg(short, long)]
    voters: Option<usize>,

    /// Number of candidates (override config)
    #[arg(short = 'C', long)]
    candidates: Option<usize>,

    /// number of candidates in a primary (RRV) election. (No primary by default.)
    #[arg(short, long)]
    primary_candidates: Option<usize>,

    // Likability factor
    #[arg(long, default_value_t = 0.4)]
    likefactor: f64,
}

fn main() {
    if let Err(err) = run() {
        println!("{}", err);
        process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    let mut config = config::Config::from_file(args.config)?;
    if let Some(ncand) = args.candidates {
        config.candidates = ncand;
    }
    if let Some(ncit) = args.voters {
        config.voters = ncit;
    }
    let ncand = config.candidates;
    let ncit = config.voters;
    let max_cand = args.primary_candidates.unwrap_or(ncand);
    let mut likability = Likability::new(args.likefactor, max_cand);
    let mut issues = MDIssue::new(
        vec![
            Issue::new(1.0, 2.0, 2.0, max_cand),
            Issue::new(0.5, 1.5, 1.5, max_cand),
        ],
        max_cand,
    );
    let mut axes: [&mut dyn Consideration; 2] = [&mut likability, &mut issues];

    let mut sim = Sim::new(ncand, ncit);

    let mut primary_sim = if let Some(pcand) = args.primary_candidates {
        Some(Sim::new(pcand, ncit))
    } else {
        None
    };

    let mut methods: Vec<MethodTracker> = config
        .methods
        .iter()
        .map(|m| MethodTracker::new(m, &sim, args.trials))
        .collect();

    run::run(
        &mut axes,
        &mut sim,
        &mut methods[..],
        args.trials,
        &args.outfile,
        &mut primary_sim,
        &config,
    )
}
