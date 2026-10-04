// © Copyright 2026 Topher Cawlfield
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::error::Error;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::considerations::Consideration;
use crate::methods::{Method, MultiWinMethod};

#[derive(Debug, Serialize, Deserialize, Clone)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
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

/// Which kind of election this config runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub enum RunMode {
    /// One winner per trial: `methods` runs (optionally behind a `primary_method`
    /// narrowing stage), each producing a single winner.
    #[default]
    #[serde(rename = "single_winner")]
    SingleWinner,
    /// A fixed-size committee per trial: each of `committee_methods` runs as the
    /// final answer (no narrowing stage), electing `committee_size` winners.
    #[serde(rename = "multi_winner")]
    MultiWinner,
}

fn default_primary() -> MultiWinMethod {
    MultiWinMethod::RRV(crate::methods::RRV {
        strat: crate::methods::Strategy::Honest,
        ranks: 25,
        k: 0.5,
        colname: crate::methods::ColName::default(),
    })
}

impl Config {
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Config, Box<dyn Error>> {
        let config_str = std::fs::read_to_string(path)?;
        Ok(Self::from_toml_str(&config_str)?)
    }

    /// Parse a config from a TOML document, as [`Config::from_file`] reads from disk.
    pub fn from_toml_str(config_str: &str) -> Result<Config, toml::de::Error> {
        toml::from_str(config_str)
    }

    /// Parse a config from a JSON document.
    ///
    /// The same shape as the TOML form -- both go through the same `serde`
    /// derives -- so this reads back what `run_sims` stores in the
    /// `voting_config` parquet metadata key, and is the convenient form for
    /// callers assembling a config programmatically (the Python bindings).
    pub fn from_json_str(config_str: &str) -> Result<Config, serde_json::Error> {
        serde_json::from_str(config_str)
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
                // The primary elects `candidates` finalists out of `primary_candidates`.
                if let Some(primary_candidates) = self.primary_candidates
                    && primary_candidates <= self.candidates
                {
                    return Err(
                        "primary_candidates must be greater than the number of candidates"
                            .to_string(),
                    );
                }
                check_unique_colnames("methods", self.methods.iter().map(Method::colname))?;
            }
            RunMode::MultiWinner => {
                if let Some(committee_size) = self.committee_size {
                    if committee_size >= self.candidates {
                        return Err(
                            "committee_size must be less than the number of candidates".to_string()
                        );
                    }
                } else {
                    return Err("mode = MultiWinner requires `committee_size`".to_string());
                }
                if self.committee_methods.is_empty() {
                    return Err(
                        "mode = MultiWinner requires at least one entry in `committee_methods`"
                            .to_string(),
                    );
                }
                check_unique_colnames(
                    "committee_methods",
                    self.committee_methods.iter().map(MultiWinMethod::colname),
                )?;
            }
        }
        Ok(())
    }
}

/// Each method's results go in the output column its `colname` names; two
/// methods sharing one would silently overwrite each other. `list` names the
/// config key the column names came from, for the error message.
fn check_unique_colnames(list: &str, colnames: impl Iterator<Item = String>) -> Result<(), String> {
    let mut first_entry: BTreeMap<String, usize> = BTreeMap::new();
    for (i, colname) in colnames.enumerate() {
        if colname.is_empty() {
            return Err(format!("`{list}` entry {} has an empty `colname`", i + 1));
        }
        if let Some(j) = first_entry.get(&colname) {
            return Err(format!(
                "`{list}` entries {} and {} would both write output column `{colname}`",
                j + 1,
                i + 1,
            ));
        }
        first_entry.insert(colname, i);
    }
    Ok(())
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
        let config: Config = toml::from_str(&base_toml(r#"mode = "multi_winner""#)).unwrap();
        assert!(config.validate().is_err());

        let toml_str = base_toml(
            r#"
            mode = "multi_winner"
            committee_size = 3
            "#,
        ) + "[[committee_methods]]\nPluralityTopN = {}\n";
        let config: Config = toml::from_str(&toml_str).unwrap();
        assert_eq!(config.mode, RunMode::MultiWinner);
        assert_eq!(config.committee_size, Some(3));
        assert!(config.validate().is_ok());
    }

    fn multi_winner_with_committee_size(committee_size: usize) -> Config {
        let toml_str = base_toml(&format!(
            "mode = \"multi_winner\"\ncommittee_size = {committee_size}"
        )) + "[[committee_methods]]\nPluralityTopN = {}\n";
        toml::from_str(&toml_str).unwrap()
    }

    #[test]
    fn committee_size_must_be_less_than_candidates() {
        // base_toml has 4 candidates.
        assert!(multi_winner_with_committee_size(3).validate().is_ok());
        for committee_size in [4, 5] {
            assert_eq!(
                multi_winner_with_committee_size(committee_size)
                    .validate()
                    .unwrap_err(),
                "committee_size must be less than the number of candidates",
                "committee_size = {committee_size}"
            );
        }
    }

    /// A single-winner config with a primary narrowing `primary_candidates`
    /// down to base_toml's 4 candidates. `extra` goes after the methods, so it
    /// can hold a `[primary_method]` table.
    fn single_winner_with_primary(primary_candidates: usize, extra: &str) -> Config {
        let toml_str = base_toml(&format!("primary_candidates = {primary_candidates}"))
            + "[[methods]]\nPlurality = { strat = \"Honest\" }\n"
            + extra;
        toml::from_str(&toml_str).unwrap()
    }

    #[test]
    fn a_single_winner_primary_validates() {
        let config = single_winner_with_primary(8, "");
        assert_eq!(config.primary_candidates, Some(8));
        assert!(config.validate().is_ok());
    }

    #[test]
    fn a_single_winner_primary_can_name_its_method() {
        let config = single_winner_with_primary(8, "[primary_method]\nPluralityTopN = {}\n");
        assert!(matches!(
            config.primary_method,
            MultiWinMethod::PluralityTopN(_)
        ));
        assert!(config.validate().is_ok());
    }

    /// The primary elects `candidates` finalists, so it needs more than that to
    /// choose from. Fewer used to panic inside RRV.
    #[test]
    fn primary_candidates_must_be_more_than_candidates() {
        for primary_candidates in [3, 4] {
            assert_eq!(
                single_winner_with_primary(primary_candidates, "")
                    .validate()
                    .unwrap_err(),
                "primary_candidates must be greater than the number of candidates",
                "primary_candidates = {primary_candidates}"
            );
        }
    }

    #[test]
    fn a_primary_still_requires_methods() {
        let config: Config = toml::from_str(&base_toml("primary_candidates = 8")).unwrap();
        assert_eq!(
            config.validate().unwrap_err(),
            "mode = SingleWinner requires at least one entry in `methods`"
        );
    }

    fn single_winner_with_methods(entries: &[&str]) -> Config {
        let mut toml_str = base_toml("");
        for entry in entries {
            toml_str += &format!("\n[[methods]]\n{entry}\n");
        }
        toml::from_str(&toml_str).unwrap()
    }

    #[test]
    fn distinct_column_names_validate() {
        let config = single_winner_with_methods(&[
            r#"Plurality = { strat = "Honest" }"#,
            r#"Plurality = { strat = "Strategic" }"#,
            r#"Range = { strat = "Honest", nranks = 10 }"#,
            r#"Range = { strat = "Honest", nranks = 2 }"#,
        ]);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn a_method_listed_twice_fails_validation() {
        let config = single_winner_with_methods(&[
            r#"Plurality = { strat = "Honest" }"#,
            r#"Borda = {}"#,
            r#"Plurality = { strat = "Honest" }"#,
        ]);
        assert_eq!(
            config.validate().unwrap_err(),
            "`methods` entries 1 and 3 would both write output column `pl_h`"
        );
    }

    /// Different methods can share a column name too: the stretch factor isn't
    /// part of STAR's, so these two would overwrite each other's results.
    #[test]
    fn different_methods_sharing_a_column_name_fail_validation() {
        let config = single_winner_with_methods(&[
            r#"STAR = { strat = "Strategic", strategic_stretch_factor = 2.0 }"#,
            r#"STAR = { strat = "Strategic", strategic_stretch_factor = 4.0 }"#,
        ]);
        let err = config.validate().unwrap_err();
        assert!(err.contains("`star_6_s`"), "{err}");
    }

    fn multi_winner_with_committee_methods(entries: &[&str]) -> Config {
        let mut toml_str = base_toml("mode = \"multi_winner\"\ncommittee_size = 2");
        for entry in entries {
            toml_str += &format!("\n[[committee_methods]]\n{entry}\n");
        }
        toml::from_str(&toml_str).unwrap()
    }

    #[test]
    fn distinct_committee_column_names_validate() {
        let config = multi_winner_with_committee_methods(&[
            "PluralityTopN = {}",
            r#"RRV = { strat = "Honest", ranks = 10, k = 0.5 }"#,
            r#"RRV = { strat = "Honest", ranks = 25, k = 0.5 }"#,
        ]);
        assert!(config.validate().is_ok());
    }

    /// `k` isn't part of RRV's column name, so these two would overwrite each
    /// other's results.
    #[test]
    fn committee_methods_sharing_a_column_name_fail_validation() {
        let config = multi_winner_with_committee_methods(&[
            "PluralityTopN = {}",
            r#"RRV = { strat = "Honest", ranks = 25, k = 0.5 }"#,
            r#"RRV = { strat = "Honest", ranks = 25, k = 1.0 }"#,
        ]);
        assert_eq!(
            config.validate().unwrap_err(),
            "`committee_methods` entries 2 and 3 would both write output column `rrv_25_h`"
        );
    }

    /// A misspelled parameter used to be dropped silently, leaving its default
    /// in place. Every config struct now rejects fields it doesn't know.
    #[test]
    fn a_misspelled_method_parameter_is_an_error() {
        let toml_str = base_toml("")
            + "[[methods]]\nRange = { strat = \"Strategic\", nranks = 10, strategic_strech_factor = 2.0 }\n";
        let err = toml::from_str::<Config>(&toml_str).unwrap_err();
        assert!(
            err.message()
                .contains("unknown field `strategic_strech_factor`"),
            "{err}"
        );
    }

    #[test]
    fn a_misspelled_consideration_parameter_is_an_error() {
        let toml_str =
            "voters = 10\ncandidates = 4\n[[considerations]]\nLikability = { maen = 0.5 }\n";
        let err = toml::from_str::<Config>(toml_str).unwrap_err();
        assert!(err.message().contains("unknown field `maen`"), "{err}");
    }

    #[test]
    fn colname_replaces_the_default_column_name() {
        let config = single_winner_with_methods(&[
            r#"Plurality = { strat = "Honest", colname = "first_choice" }"#,
            // The table form, as in configs/range_strat.toml.
            "[methods.Range]\nstrat = \"Honest\"\nnranks = 10\ncolname = \"score_10\"",
            r#"Borda = {}"#,
        ]);
        let colnames: Vec<String> = config.methods.iter().map(Method::colname).collect();
        assert_eq!(colnames, ["first_choice", "score_10", "Borda_h"]);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn a_misspelled_colname_is_an_error() {
        let toml_str =
            base_toml("") + "[[methods]]\nPlurality = { strat = \"Honest\", colnmae = \"x\" }\n";
        let err = toml::from_str::<Config>(&toml_str).unwrap_err();
        assert!(err.message().contains("unknown field `colnmae`"), "{err}");
    }

    /// The motivating case: a sweep over a parameter that isn't part of the
    /// default column name, told apart by giving each its own.
    #[test]
    fn distinct_colnames_resolve_a_default_name_collision() {
        let config = single_winner_with_methods(&[
            r#"STAR = { strat = "Strategic", strategic_stretch_factor = 2.0, colname = "star_s_2x" }"#,
            r#"STAR = { strat = "Strategic", strategic_stretch_factor = 4.0, colname = "star_s_4x" }"#,
        ]);
        assert!(config.validate().is_ok());

        let committees = multi_winner_with_committee_methods(&[
            r#"RRV = { strat = "Honest", ranks = 25, k = 0.5, colname = "rrv_k05" }"#,
            r#"RRV = { strat = "Honest", ranks = 25, k = 1.0, colname = "rrv_k10" }"#,
        ]);
        assert!(committees.validate().is_ok());
    }

    #[test]
    fn a_colname_colliding_with_another_methods_default_fails_validation() {
        let config = single_winner_with_methods(&[
            r#"Plurality = { strat = "Honest" }"#,
            r#"Borda = { colname = "pl_h" }"#,
        ]);
        assert_eq!(
            config.validate().unwrap_err(),
            "`methods` entries 1 and 2 would both write output column `pl_h`"
        );
    }

    #[test]
    fn an_empty_colname_fails_validation() {
        let config =
            single_winner_with_methods(&[r#"Plurality = { strat = "Honest", colname = "" }"#]);
        assert_eq!(
            config.validate().unwrap_err(),
            "`methods` entry 1 has an empty `colname`"
        );
    }

    /// The config is stored as JSON in the parquet metadata: a colname must
    /// survive the trip, and an unset one must not appear at all.
    #[test]
    fn colname_round_trips_through_json_and_is_omitted_when_unset() {
        let config = single_winner_with_methods(&[
            r#"Plurality = { strat = "Honest" }"#,
            r#"Plurality = { strat = "Strategic", colname = "pl_tactical" }"#,
        ]);
        let json = serde_json::to_string(&config).unwrap();
        assert!(
            json.contains(r#"{"Plurality":{"strat":"Honest"}}"#),
            "{json}"
        );
        let back = Config::from_json_str(&json).unwrap();
        let colnames: Vec<String> = back.methods.iter().map(Method::colname).collect();
        assert_eq!(colnames, ["pl_h", "pl_tactical"]);
    }
}
