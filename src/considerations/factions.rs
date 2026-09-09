// © Copyright 2025 Topher Cawlfield
// SPDX-License-Identifier: Apache-2.0

use std::arch::x86_64::_MM_EXCEPT_UNDERFLOW;

use super::ConsiderationSim;
use crate::sim::Sim;
use ndarray::Array2;
use rand::{Rng, RngExt as _};
use rand_distr::StandardNormal;

/// A Factions consideration is an N-dimensional issue space (utilities fall off
/// with the Euclidean distance between a voter and a candidate), but instead of
/// defining each axis separately, the config lists factions -- each a cluster of
/// voters and candidates.
///
/// Each candidate belongs to exactly one faction and carries two utility
/// bonuses: `universal_likability_ceiling` (added to every voter's utility for
/// that candidate) and `in_group_likability_ceiling` (added on top, only for
/// voters in the candidate's own faction). Both are drawn `U(0, ceiling)` once
/// per candidate.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Factions {
    /// Dimensionality of the issue space. Every `*_center` must be this long.
    pub dimensions: usize,
    /// How a voter/choice distance becomes a perceived utility.
    pub distance_scaling: DistanceFunction,
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
    /// Upper bound of the per-candidate likability bonus every voter sees.
    #[serde(default)]
    pub universal_likability_ceiling: f64,
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
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum DistanceFunction {
    /// Perceived utility is the negative of the Euclidean distance between the voter and
    /// candidate in the issue space.
    NegativeEuclidean,
    /// Perceived utility is 1/(1 + sigma x^2) where x is the Euclidean distance between the voter
    /// and candidate in the issue space.
    QGaussian2(f64),
}

impl DistanceFunction {
    /// Utility for a given *squared* voter/choice distance.
    fn utility(&self, dist_sq: f64) -> f64 {
        match self {
            DistanceFunction::NegativeEuclidean => -dist_sq.sqrt(),
            DistanceFunction::QGaussian2(sigma) => 1.0 / (1.0 + sigma * dist_sq),
        }
    }
}

#[derive(Debug)]
pub struct FactionsSim {
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
}

impl Factions {
    pub fn new_sim(&self, sim: &Sim) -> FactionsSim {
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

        assert!(
            self.factions.iter().all(|f| f.popularity > 0.0),
            "All factions require positive popularity"
        );
        let popularity_cdf: Vec<f64> = self
            .factions
            .iter()
            .scan(0.0, |cdf, f| Some(*cdf + f.popularity))
            .collect();
        let ttl_popularity = *popularity_cdf.last().unwrap();

        FactionsSim {
            dims: self.dimensions,
            distance_scaling: self.distance_scaling.clone(),
            factions: self.factions.clone(),
            popularity_cdf,
            total_popularity: ttl_popularity,
            cand_positions: Array2::zeros((sim.ncand, self.dimensions)),
            cand_factions: vec![0; sim.ncand],
        }
    }
}

impl ConsiderationSim for FactionsSim {
    // See the note on IssuesSim::add_to_scores: out of line on purpose.
    #[inline(never)]
    fn add_to_scores<R: Rng + ?Sized>(&mut self, scores: &mut Array2<f64>, rng: &mut R) {
        let (_nvtr, ncand) = scores.dim();
        let nfactions = self.factions.len();

        // Candidates: round-robin from a random starting faction. Each also draws
        // its universal and in-group likability bonuses.
        let start = rng.random_range(0..nfactions);
        let mut universal_like = vec![0.0f64; ncand];
        let mut in_group_like = vec![0.0f64; ncand];
        for ichoice in 0..ncand {
            let ifac = (start + ichoice) % nfactions;
            self.cand_factions[ichoice] = ifac;
            let faction = &self.factions[ifac];
            let center = faction.candidate_center();
            let spread = faction.candidate_spread();
            for (pos, &c) in self.cand_positions.row_mut(ichoice).iter_mut().zip(center) {
                let z: f64 = rng.sample(StandardNormal);
                *pos = c + z * spread;
            }
            universal_like[ichoice] = draw_ceiling(rng, faction.universal_likability_ceiling);
            in_group_like[ichoice] = draw_ceiling(rng, faction.in_group_likability_ceiling);
        }
        log::debug!("faction choice positions: {:?}", self.cand_positions);

        // Voters: each is assigned a faction weighted by popularity, drawn around
        // that faction's center, then scored against every candidate.
        let mut vtr_pos = vec![0.0f64; self.dims];
        for mut vtr_scores in scores.rows_mut() {
            let r = rng.random_range(0.0..self.total_popularity);
            let vfac = self.popularity_cdf.partition_point(|&c| c <= r);
            let faction = &self.factions[vfac];
            for (pos, &c) in vtr_pos.iter_mut().zip(&faction.voter_center) {
                let z: f64 = rng.sample(StandardNormal);
                *pos = c + z * faction.voter_spread;
            }
            for ichoice in 0..ncand {
                let dist_sq: f64 = self
                    .cand_positions
                    .row(ichoice)
                    .iter()
                    .zip(&vtr_pos)
                    .map(|(&cp, &vp)| (cp - vp).powi(2))
                    .sum();
                let mut utility = self.distance_scaling.utility(dist_sq);
                utility += universal_like[ichoice];
                if self.cand_factions[ichoice] == vfac {
                    utility += in_group_like[ichoice];
                }
                vtr_scores[ichoice] += utility;
            }
        }
    }

    fn get_dim(&self) -> usize {
        self.dims
    }

    fn get_name(&self) -> String {
        "factions".to_string()
    }

    fn push_posn_elements(&self, report: &mut dyn FnMut(f64, bool), final_choices: &[usize]) {
        for &fc in final_choices {
            for d in 0..self.dims {
                report(self.cand_positions[(fc, d)], d == self.dims - 1);
            }
        }
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
            universal_likability_ceiling: 0.0,
            in_group_likability_ceiling: 0.0,
        }
    }

    #[test]
    fn distance_function_utilities() {
        // NegativeEuclidean takes the sqrt of the squared distance.
        assert_eq!(DistanceFunction::NegativeEuclidean.utility(9.0), -3.0);
        // QGaussian2: 1 / (1 + sigma * dist_sq)
        assert_eq!(DistanceFunction::QGaussian2(0.0).utility(100.0), 1.0);
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
        Factions {
            dimensions: 2,
            distance_scaling: DistanceFunction::NegativeEuclidean,
            factions: vec![],
        }
        .new_sim(&Sim::new(3, 5));
    }

    #[test]
    #[should_panic(expected = "wrong length")]
    fn new_sim_rejects_mismatched_center_length() {
        Factions {
            dimensions: 3,
            distance_scaling: DistanceFunction::NegativeEuclidean,
            factions: vec![faction([0.0, 0.0], 1.0)], // 2 coords, dimensions = 3
        }
        .new_sim(&Sim::new(3, 5));
    }

    #[test]
    #[should_panic(expected = "positive total popularity")]
    fn new_sim_rejects_zero_total_popularity() {
        Factions {
            dimensions: 2,
            distance_scaling: DistanceFunction::NegativeEuclidean,
            factions: vec![faction([0.0, 0.0], 0.0), faction([1.0, 1.0], 0.0)],
        }
        .new_sim(&Sim::new(3, 5));
    }

    #[test]
    fn scores_are_negative_distance_when_spreads_and_likability_are_zero() {
        // One faction, voters pinned to the center (spread 0) but candidates
        // scattered. Every voter should score candidate c at -||choice_pos[c]||.
        let params = Factions {
            dimensions: 2,
            distance_scaling: DistanceFunction::NegativeEuclidean,
            factions: vec![Faction {
                voter_spread: 0.0,
                candidate_spread: Some(3.0),
                ..faction([0.0, 0.0], 1.0)
            }],
        };
        let mut sim = params.new_sim(&Sim::new(4, 6));
        let mut scores = Array2::zeros((6, 4));
        sim.add_to_scores(&mut scores, &mut StdRng::seed_from_u64(1));

        for ichoice in 0..4 {
            let (x, y) = (
                sim.cand_positions[(ichoice, 0)],
                sim.cand_positions[(ichoice, 1)],
            );
            let expected = -(x * x + y * y).sqrt();
            for ivtr in 0..6 {
                assert_abs_diff_eq!(scores[(ivtr, ichoice)], expected, epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn in_group_likability_only_benefits_own_faction_choices() {
        // Both factions share a center (distance contributes nothing) and all
        // spreads are 0, so the only signal is in-group likability. Popularity
        // [1, 0] puts every voter in faction 0.
        let params = Factions {
            dimensions: 2,
            distance_scaling: DistanceFunction::NegativeEuclidean,
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

        for ichoice in 0..4 {
            let in_faction_0 = sim.cand_factions[ichoice] == 0;
            for ivtr in 0..8 {
                let s = scores[(ivtr, ichoice)];
                if in_faction_0 {
                    assert!((0.0..=5.0).contains(&s), "faction-0 choice score {s}");
                } else {
                    assert_eq!(s, 0.0, "other-faction choice score");
                }
            }
        }
        // With 4 candidates round-robin over 2 factions, faction 0 gets 2 of them.
        assert_eq!(sim.cand_factions.iter().filter(|&&f| f == 0).count(), 2);
    }

    #[test]
    fn seeded_rng_makes_scores_reproducible() {
        let params = Factions {
            dimensions: 2,
            distance_scaling: DistanceFunction::QGaussian2(0.5),
            factions: vec![
                Faction {
                    voter_spread: 1.0,
                    universal_likability_ceiling: 0.5,
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
