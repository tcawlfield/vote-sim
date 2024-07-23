use std::error::Error;

use clap::Parser;
use std::process;

// Local libraries
mod consideration;
mod cov_matrix;
mod methods;
mod run;
mod sim;

use consideration::*;
use methods::*;
use sim::Sim;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Number of voters
    #[arg(short, long, default_value_t = 7)]
    voters: usize,

    /// Number of candidates
    #[arg(short, long, default_value_t = 4)]
    candidates: usize,

    /// number of candidates in a primary (RRV) election. (No primary by default.)
    #[arg(short, long)]
    primary_candidates: Option<usize>,

    /// Number of trials
    #[arg(short, long, default_value_t = 1)]
    trials: usize,

    /// Output file
    #[arg(short, long)]
    outfile: Option<std::ffi::OsString>,

    // Likability factor
    #[arg(long, default_value_t=0.4)]
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

    let ncand = args.candidates;
    let ncit = args.voters;
    let max_cand = args.primary_candidates.unwrap_or(ncand);
    let mut likability = Likability::new(args.likefactor, max_cand);
    let mut issues = MDIssue::new(
        vec![
            Issue::new(1.0, 1.0, 1.0, max_cand),
            Issue::new(0.5, 0.0, 0.0, max_cand),
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

    let mut methods: Vec<MethodTracker> = vec![
        MethodTracker::new(
            Box::new(Plurality::new(&sim, Strategy::Honest)),
            args.trials,
        ),
        MethodTracker::new(
            Box::new(Plurality::new(&sim, Strategy::Strategic)),
            args.trials,
        ),
        MethodTracker::new(
            Box::new(RangeVoting::new(&sim, 10, Strategy::Honest)),
            args.trials,
        ),
        MethodTracker::new(
            Box::new(RangeVoting::new(&sim, 10, Strategy::Strategic)),
            args.trials,
        ),
        MethodTracker::new(
            Box::new(RangeVoting::new(&sim, 2, Strategy::Honest)),
            args.trials,
        ),
        MethodTracker::new(
            Box::new(RangeVoting::new(&sim, 2, Strategy::Strategic)),
            args.trials,
        ),
    ];

    run::run(
        &mut axes,
        &mut sim,
        &mut methods[..],
        args.trials,
        &args.outfile,
        &mut primary_sim,
    )
}
