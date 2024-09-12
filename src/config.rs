use std::error::Error;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::considerations::Consideration;
use crate::methods::{Method, MultiWinMethod};

#[derive(Debug, Serialize, Deserialize)]
pub struct Config {
    pub voters: usize,
    pub candidates: usize,
    pub primary_candidates: Option<usize>,
    pub considerations: Vec<Consideration>,
    pub methods: Vec<Method>,
    pub primary_method: MultiWinMethod,
}

impl Config {
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Config, Box<dyn Error>> {
        let config_str = std::fs::read_to_string(path)?;
        // let file = File::open(path)?;
        // let reader = BufReader::new(file);

        // Read the TOML contents of the file as an instance of `Config`.
        let config = toml::from_str(&config_str)?;

        // Return the `User`.
        Ok(config)
    }
}
