// © Copyright 2026 Topher Cawlfield
// SPDX-License-Identifier: Apache-2.0

//! Defines Electorate and friends

use super::ConsiderationSim;
use crate::{out_types::ElectorateInfo, sim::Sim};
use ndarray::Array2;
use rand::{Rng, RngExt as _};
use rand_distr::StandardNormal;

/// An Electorate consideration is an N-dimensional issue space in which voters and candidates
/// are divided into factions. Each faction has a center and a spread (isotropic Gaussian) for
/// both voters and candidates. Voter utility is a function of the distance between a voter
/// and candidate in the issue space, optionally with a special in-group likability bonus for
/// candidates in the voter's own faction.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Electorate {
    /// Dimensionality of the issue space. Every `*_center` must be this long.
    pub dimensions: usize,
    /// How a voter/candidate distance becomes a perceived utility.
    #[serde(default)]
    pub distance_function: DistanceFunction,
    /// List of the separate factions.
    pub factions: Vec<Faction>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Faction {
    /// Relative weight for assigning voters to this faction (need not sum to 1
    /// across factions). Candidates are assigned round-robin regardless.
    pub popularity: f64,
    /// Center of this faction's voters in issue space.
    pub voter_center: Vec<f64>,
    /// Center of this faction's candidates. Defaults to `voter_center`.
    #[serde(default)]
    pub candidate_center: Option<Vec<f64>>,
    /// Std-dev of the (isotropic Gaussian) scatter of voters around the center.
    pub voter_spread: f64,
    /// Scatter of candidates around their center. Defaults to `voter_spread`.
    #[serde(default)]
    pub candidate_spread: Option<f64>,
    /// Upper bound of the extra per-candidate likability bonus that only voters
    /// in the candidate's own faction see (added on top of the universal one).
    #[serde(default)]
    pub in_group_likability_ceiling: f64,
}

impl Faction {
    fn candidate_center(&self) -> &[f64] {
        self.candidate_center
            .as_deref()
            .unwrap_or(&self.voter_center)
    }

    fn candidate_spread(&self) -> f64 {
        self.candidate_spread.unwrap_or(self.voter_spread)
    }
}

/// Scaling function for the distance between a voter and candidate in the issue space.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, Default, PartialEq)]
pub enum DistanceFunction {
    /// Perceived utility is the negative of the Euclidean distance between the voter and
    /// candidate in the issue space.
    #[default]
    #[serde(alias = "negative_euclidean")]
    NegativeEuclidean,
    /// Perceived utility is 1/(1 + x^2 / sigma^2) where x is the Euclidean distance between the voter
    /// and candidate in the issue space.
    #[serde(alias = "q_gaussian_2")]
    QGaussian2(f64),
}

impl DistanceFunction {
    /// Utility for a given *squared* voter/candidate distance.
    fn utility(&self, dist_sq: f64) -> f64 {
        match self {
            DistanceFunction::NegativeEuclidean => -dist_sq.sqrt(),
            DistanceFunction::QGaussian2(sigma) => 1.0 / (1.0 + dist_sq / (sigma * sigma)),
        }
    }
}

#[derive(Debug)]
pub struct ElectorateSim {
    dims: usize,
    distance_scaling: DistanceFunction,
    factions: Vec<Faction>,
    /// Inclusive cumulative sum of `popularity`; last entry is `total_popularity`.
    popularity_cdf: Vec<f64>,
    total_popularity: f64,
    /// Candidate positions in issue space (`ncand` x `dims`), for reporting.
    cand_positions: Array2<f64>,
    /// Faction index of each candidate.
    cand_factions: Vec<usize>,
    /// in-group likability bonus for each candidate
    in_group_like: Vec<f64>,
    /// Position of some voter in issue space (scratch vector, reused for every voter).
    vtr_pos: Vec<f64>,
}

impl Electorate {
    pub fn new_sim(&self, sim: &Sim) -> ElectorateSim {
        assert!(
            !self.factions.is_empty(),
            "Factions consideration needs at least one faction"
        );
        assert!(self.dimensions > 0, "Factions needs dimensions > 0");
        for (i, faction) in self.factions.iter().enumerate() {
            assert_eq!(
                faction.voter_center.len(),
                self.dimensions,
                "faction {i}: voter_center has the wrong length"
            );
            if let Some(center) = &faction.candidate_center {
                assert_eq!(
                    center.len(),
                    self.dimensions,
                    "faction {i}: candidate_center has the wrong length"
                );
            }
            assert!(
                faction.popularity >= 0.0,
                "faction {i}: popularity must be non-negative"
            );
        }

        let popularity_cdf: Vec<f64> = self
            .factions
            .iter()
            .scan(0.0, |cdf, f| {
                *cdf += f.popularity;
                Some(*cdf)
            })
            .collect();
        let ttl_popularity = *popularity_cdf.last().unwrap();
        assert!(
            ttl_popularity > 0.0,
            "Factions needs positive total popularity"
        );

        ElectorateSim {
            dims: self.dimensions,
            distance_scaling: self.distance_function.clone(),
            factions: self.factions.clone(),
            popularity_cdf,
            total_popularity: ttl_popularity,
            cand_positions: Array2::zeros((sim.ncand, self.dimensions)),
            cand_factions: vec![0; sim.ncand],
            in_group_like: vec![0.0f64; sim.ncand],
            vtr_pos: vec![0.0f64; self.dimensions],
        }
    }
}

impl ConsiderationSim for ElectorateSim {
    // See the note on IssuesSim::add_to_scores: out of line on purpose.
    #[inline(never)]
    fn add_to_scores<R: Rng + ?Sized>(&mut self, scores: &mut Array2<f64>, rng: &mut R) {
        let (nvtr, ncand) = scores.dim();
        let nfactions = self.factions.len();

        // Candidates: round-robin from a random starting faction. Each also draws
        // its in-group likability bonuses.
        let start = rng.random_range(0..nfactions);
        for icand in 0..ncand {
            let ifac = (start + icand) % nfactions;
            self.cand_factions[icand] = ifac;
            let faction = &self.factions[ifac];
            let center = faction.candidate_center();
            let spread = faction.candidate_spread();
            for (pos, &c) in self.cand_positions.row_mut(icand).iter_mut().zip(center) {
                let z: f64 = rng.sample(StandardNormal);
                *pos = c + z * spread;
            }
            self.in_group_like[icand] = draw_ceiling(rng, faction.in_group_likability_ceiling);
        }
        log::debug!("candidate faction numbers: {:?}", self.cand_factions);
        log::debug!("faction candidate positions:\n{}", self.cand_positions);

        // Voters: each is assigned a faction weighted by popularity, drawn around
        // that faction's center, then scored against every candidate.
        for mut vtr_scores in scores.rows_mut() {
            let r = rng.random_range(0.0..self.total_popularity);
            let vfac = self.popularity_cdf.partition_point(|&c| c <= r);
            let faction = &self.factions[vfac];
            for (pos, &c) in self.vtr_pos.iter_mut().zip(&faction.voter_center) {
                let z: f64 = rng.sample(StandardNormal);
                *pos = c + z * faction.voter_spread;
            }
            if nvtr < 50 {
                log::debug!("Voter faction={} pos={:?}", vfac, self.vtr_pos);
            }
            for icand in 0..ncand {
                let dist_sq: f64 = self
                    .cand_positions
                    .row(icand)
                    .iter()
                    .zip(&self.vtr_pos)
                    .map(|(&cp, &vp)| (cp - vp).powi(2))
                    .sum();
                let mut utility = self.distance_scaling.utility(dist_sq);
                if self.cand_factions[icand] == vfac {
                    utility += self.in_group_like[icand];
                }
                vtr_scores[icand] += utility;
            }
        }
    }

    fn get_dim(&self) -> usize {
        self.dims
    }

    fn get_name(&self) -> String {
        "electorate".to_string()
    }

    fn push_posn_elements(&self, report: &mut dyn FnMut(f64, bool), final_candidates: &[usize]) {
        for &fc in final_candidates {
            for d in 0..self.dims {
                report(self.cand_positions[(fc, d)], d == self.dims - 1);
            }
        }
    }
}

impl ElectorateSim {
    pub fn make_faction_info(
        &self,
        premade_positions: Vec<Vec<f64>>,
        final_candidates: &[usize],
    ) -> ElectorateInfo {
        let mut fi = ElectorateInfo {
            positions: premade_positions,
            faction: Vec::with_capacity(final_candidates.len()),
            in_group_likability: Vec::with_capacity(final_candidates.len()),
        };
        for &fc in final_candidates {
            fi.faction.push(self.cand_factions[fc] as u32);
            fi.in_group_likability.push(self.in_group_like[fc]);
        }
        fi
    }
}

/// A `U(0, ceiling)` draw, or 0 when the ceiling is not positive.
fn draw_ceiling<R: Rng + ?Sized>(rng: &mut R, ceiling: f64) -> f64 {
    if ceiling > 0.0 {
        rng.random_range(0.0..ceiling)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn faction(center: [f64; 2], popularity: f64) -> Faction {
        Faction {
            popularity,
            voter_center: center.to_vec(),
            candidate_center: None,
            voter_spread: 0.0,
            candidate_spread: None,
            in_group_likability_ceiling: 0.0,
        }
    }

    #[test]
    fn distance_function_utilities() {
        // NegativeEuclidean takes the sqrt of the squared distance.
        assert_eq!(DistanceFunction::NegativeEuclidean.utility(9.0), -3.0);
        // QGaussian2: 1 / (1 + dist_sq / sigma^2))
        assert_eq!(DistanceFunction::QGaussian2(10.0).utility(100.0), 0.5);
        assert_eq!(DistanceFunction::QGaussian2(1.0).utility(3.0), 0.25);
    }

    #[test]
    fn candidate_geometry_falls_back_to_voter_geometry() {
        let mut f = faction([1.0, 2.0], 1.0);
        f.voter_spread = 0.7;
        assert_eq!(f.candidate_center(), [1.0, 2.0]);
        assert_eq!(f.candidate_spread(), 0.7);

        f.candidate_center = Some(vec![9.0, 9.0]);
        f.candidate_spread = Some(0.1);
        assert_eq!(f.candidate_center(), [9.0, 9.0]);
        assert_eq!(f.candidate_spread(), 0.1);
    }

    #[test]
    #[should_panic(expected = "at least one faction")]
    fn new_sim_rejects_no_factions() {
        Electorate {
            dimensions: 2,
            distance_function: DistanceFunction::NegativeEuclidean,
            factions: vec![],
        }
        .new_sim(&Sim::new(3, 5));
    }

    #[test]
    #[should_panic(expected = "wrong length")]
    fn new_sim_rejects_mismatched_center_length() {
        Electorate {
            dimensions: 3,
            distance_function: DistanceFunction::NegativeEuclidean,
            factions: vec![faction([0.0, 0.0], 1.0)], // 2 coords, dimensions = 3
        }
        .new_sim(&Sim::new(3, 5));
    }

    #[test]
    #[should_panic(expected = "positive total popularity")]
    fn new_sim_rejects_zero_total_popularity() {
        Electorate {
            dimensions: 2,
            distance_function: DistanceFunction::NegativeEuclidean,
            factions: vec![faction([0.0, 0.0], 0.0), faction([1.0, 1.0], 0.0)],
        }
        .new_sim(&Sim::new(3, 5));
    }

    #[test]
    fn scores_are_negative_distance_when_spreads_and_likability_are_zero() {
        // One faction, voters pinned to the center (spread 0) but candidates
        // scattered. Every voter should score candidate c at -||candidate_pos[c]||.
        let params = Electorate {
            dimensions: 2,
            distance_function: DistanceFunction::NegativeEuclidean,
            factions: vec![Faction {
                voter_spread: 0.0,
                candidate_spread: Some(3.0),
                ..faction([0.0, 0.0], 1.0)
            }],
        };
        let mut sim = params.new_sim(&Sim::new(4, 6));
        let mut scores = Array2::zeros((6, 4));
        sim.add_to_scores(&mut scores, &mut StdRng::seed_from_u64(1));

        for icand in 0..4 {
            let (x, y) = (
                sim.cand_positions[(icand, 0)],
                sim.cand_positions[(icand, 1)],
            );
            let expected = -(x * x + y * y).sqrt();
            for ivtr in 0..6 {
                assert_abs_diff_eq!(scores[(ivtr, icand)], expected, epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn in_group_likability_only_benefits_own_faction_candidates() {
        // Both factions share a center (distance contributes nothing) and all
        // spreads are 0, so the only signal is in-group likability. Popularity
        // [1, 0] puts every voter in faction 0.
        let params = Electorate {
            dimensions: 2,
            distance_function: DistanceFunction::NegativeEuclidean,
            factions: vec![
                Faction {
                    in_group_likability_ceiling: 5.0,
                    ..faction([0.0, 0.0], 1.0)
                },
                Faction {
                    in_group_likability_ceiling: 5.0,
                    ..faction([0.0, 0.0], 0.0)
                },
            ],
        };
        let mut sim = params.new_sim(&Sim::new(4, 8));
        let mut scores = Array2::zeros((8, 4));
        sim.add_to_scores(&mut scores, &mut StdRng::seed_from_u64(7));

        for icand in 0..4 {
            let in_faction_0 = sim.cand_factions[icand] == 0;
            for ivtr in 0..8 {
                let s = scores[(ivtr, icand)];
                if in_faction_0 {
                    assert!((0.0..=5.0).contains(&s), "faction-0 candidate score {s}");
                } else {
                    assert_eq!(s, 0.0, "other-faction candidate score");
                }
            }
        }
        // With 4 candidates round-robin over 2 factions, faction 0 gets 2 of them.
        assert_eq!(sim.cand_factions.iter().filter(|&&f| f == 0).count(), 2);
    }

    #[test]
    fn seeded_rng_makes_scores_reproducible() {
        let params = Electorate {
            dimensions: 2,
            distance_function: DistanceFunction::QGaussian2(0.5),
            factions: vec![
                Faction {
                    voter_spread: 1.0,
                    in_group_likability_ceiling: 1.0,
                    ..faction([0.0, 0.0], 3.0)
                },
                Faction {
                    voter_spread: 0.8,
                    ..faction([2.0, -1.0], 1.0)
                },
            ],
        };
        let run = || {
            let mut sim = params.new_sim(&Sim::new(5, 40));
            let mut scores = Array2::zeros((40, 5));
            sim.add_to_scores(&mut scores, &mut StdRng::seed_from_u64(0xF00D));
            scores
        };
        assert_eq!(run(), run());
    }
}
