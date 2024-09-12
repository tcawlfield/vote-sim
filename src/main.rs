use std::error::Error;
use std::ffi::OsString;

use clap::Parser;
use std::process;

// Local libraries
mod config;
mod considerations;
mod cov_matrix;
mod method_tracker;
mod methods;
mod run;
mod sim;

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
    if let Some(pcand) = args.primary_candidates {
        config.primary_candidates = Some(pcand);
    }

    run::run(&config, args.trials, &args.outfile)
}
