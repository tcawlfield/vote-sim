// © Copyright 2026 Topher Cawlfield
// SPDX-License-Identifier: Apache-2.0

use std::error::Error;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::considerations::Consideration;
use crate::methods::{Method, MultiWinMethod};

/// Which kind of election this config runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum RunMode {
    /// One winner per trial: `methods` runs (optionally behind a `primary_method`
    /// narrowing stage), each producing a single winner.
    #[default]
    SingleWinner,
    /// A fixed-size committee per trial: each of `committee_methods` runs as the
    /// final answer (no narrowing stage), electing `committee_size` winners.
    MultiWinner,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub voters: usize,
    pub candidates: usize,
    pub primary_candidates: Option<usize>,
    pub considerations: Vec<Consideration>,
    #[serde(default)]
    pub mode: RunMode,
    /// Single-winner methods. Used when `mode == SingleWinner`.
    #[serde(default)]
    pub methods: Vec<Method>,
    #[serde(default = "default_primary")]
    pub primary_method: MultiWinMethod,
    /// Committee size. Required when `mode == MultiWinner`.
    pub committee_size: Option<usize>,
    /// Multi-winner methods run as the final election. Used when
    /// `mode == MultiWinner`.
    #[serde(default)]
    pub committee_methods: Vec<MultiWinMethod>,
}

fn default_primary() -> MultiWinMethod {
    MultiWinMethod::RRV(crate::methods::RRV {
        strat: crate::methods::Strategy::Honest,
        ranks: 25,
        k: 0.5,
    })
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

    /// Check that the fields relevant to `mode` are actually populated.
    pub fn validate(&self) -> Result<(), String> {
        match self.mode {
            RunMode::SingleWinner => {
                if self.methods.is_empty() {
                    return Err(
                        "mode = SingleWinner requires at least one entry in `methods`".to_string(),
                    );
                }
            }
            RunMode::MultiWinner => {
                if self.committee_size.is_none() {
                    return Err("mode = MultiWinner requires `committee_size`".to_string());
                }
                if self.committee_methods.is_empty() {
                    return Err(
                        "mode = MultiWinner requires at least one entry in `committee_methods`"
                            .to_string(),
                    );
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // `extra` (scalar `key = value` lines) must come before the trailing
    // `[[considerations]]` table -- TOML would otherwise fold it into that
    // array-of-tables entry.
    fn base_toml(extra: &str) -> String {
        format!(
            r#"
            voters = 10
            candidates = 4
            {extra}
            [[considerations]]
            Likability = {{ mean = 0.5 }}
            "#
        )
    }

    #[test]
    fn defaults_to_single_winner_mode() {
        let toml_str = format!(
            "{}\n[[methods]]\nPlurality = {{ strat = \"Honest\" }}\n",
            base_toml("")
        );
        let config: Config = toml::from_str(&toml_str).unwrap();
        assert_eq!(config.mode, RunMode::SingleWinner);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn single_winner_mode_requires_methods() {
        let config: Config = toml::from_str(&base_toml("")).unwrap();
        assert_eq!(config.mode, RunMode::SingleWinner);
        assert!(config.validate().is_err());
    }

    #[test]
    fn multi_winner_mode_requires_committee_size_and_methods() {
        let config: Config = toml::from_str(&base_toml(r#"mode = "MultiWinner""#)).unwrap();
        assert!(config.validate().is_err());

        let toml_str = base_toml(
            r#"
            mode = "MultiWinner"
            committee_size = 3
            "#,
        ) + "[[committee_methods]]\nPluralityTopN = {}\n";
        let config: Config = toml::from_str(&toml_str).unwrap();
        assert_eq!(config.mode, RunMode::MultiWinner);
        assert_eq!(config.committee_size, Some(3));
        assert!(config.validate().is_ok());
    }
}
