// © Copyright 2026 Topher Cawlfield
// SPDX-License-Identifier: Apache-2.0

//! Defines Electorate and friends

use super::ConsiderationSim;
use crate::{out_types::ElectorateInfo, sim::Sim};
use ndarray::{Array2, ArrayViewMut1};
use rand::{Rng, RngExt as _};
use rand_distr::StandardNormal;

/// An Electorate consideration is an N-dimensional issue space in which voters and candidates
/// are divided into factions. Each faction has a center and a spread for both voters and
/// candidates: voters scatter as an isotropic Gaussian, candidates as one too or uniformly
/// (see `candidate_distribution`). Voter utility is a function of the distance between a voter
/// and candidate in the issue space, optionally with a special in-group likability bonus for
/// candidates in the voter's own faction.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct Electorate {
    /// Dimensionality of the issue space. Every `*_center` must be this long.
    pub dimensions: usize,
    /// How a voter/candidate distance becomes a perceived utility.
    #[serde(default)]
    pub distance_function: DistanceFunction,
    /// How candidates scatter around their faction's center: Gaussian (the
    /// default), or uniformly within `candidate_spread` of it.
    #[serde(default)]
    pub candidate_distribution: PositionDistribution,
    /// List of the separate factions.
    pub factions: Vec<Faction>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
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
    /// Scatter of candidates around their center: the standard deviation for a
    /// Gaussian `candidate_distribution`, the maximum distance for a uniform
    /// one. Defaults to `voter_spread`.
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
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
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

/// How positions scatter around a faction's center, `spread` setting the scale.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize, Default, PartialEq)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub enum PositionDistribution {
    /// Each coordinate normal around the center, `spread` its standard deviation.
    #[default]
    #[serde(alias = "gaussian")]
    Gaussian,
    /// Uniform over the ball of radius `spread` around the center: an interval
    /// in 1-D, a disk in 2-D.
    ///
    /// Drawn by rejection: a point uniform in the surrounding cube, drawn
    /// again until it falls in the ball. The share kept falls fast with
    /// dimension -- 79% in 2-D, 52% in 3-D, but 16% in 5-D and 0.25% in 10-D --
    /// so beyond 3 dimensions prefer Gaussian. (Scaling a Gaussian vector to
    /// length `spread * U^(1/D)` would be exact in any dimension, if that's
    /// ever needed.)
    #[serde(alias = "uniform")]
    Uniform,
}

impl PositionDistribution {
    /// Draw a position around `center` into `pos`.
    fn draw<R: Rng + ?Sized>(
        &self,
        rng: &mut R,
        center: &[f64],
        spread: f64,
        mut pos: ArrayViewMut1<f64>,
    ) {
        match self {
            PositionDistribution::Gaussian => {
                for (p, &c) in pos.iter_mut().zip(center) {
                    let z: f64 = rng.sample(StandardNormal);
                    *p = c + z * spread;
                }
            }
            PositionDistribution::Uniform => {
                // An offset uniform in the unit cube, until it's in the unit ball.
                loop {
                    let mut dist_sq = 0.0;
                    for p in pos.iter_mut() {
                        let x: f64 = rng.random_range(-1.0..=1.0);
                        *p = x;
                        dist_sq += x * x;
                    }
                    if dist_sq <= 1.0 {
                        break;
                    }
                }
                for (p, &c) in pos.iter_mut().zip(center) {
                    *p = c + *p * spread;
                }
            }
        }
    }
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
    cand_distribution: PositionDistribution,
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
    /// How many voters were drawn into each faction this trial.
    faction_voters: Vec<u32>,
    /// Position of some voter in issue space (scratch vector, reused for every voter).
    vtr_pos: Vec<f64>,
}

impl Electorate {
    /// Check that every faction fits the issue space and that voters can be
    /// assigned to factions at all.
    pub fn validate(&self) -> Result<(), String> {
        if self.dimensions == 0 {
            return Err("Electorate needs dimensions > 0".to_string());
        }
        if self.factions.is_empty() {
            return Err("Electorate needs at least one faction".to_string());
        }
        for (i, faction) in self.factions.iter().enumerate() {
            let n = i + 1;
            if faction.voter_center.len() != self.dimensions {
                return Err(format!(
                    "Electorate faction {n}: voter_center has length {}, but dimensions = {}",
                    faction.voter_center.len(),
                    self.dimensions
                ));
            }
            if let Some(center) = &faction.candidate_center
                && center.len() != self.dimensions
            {
                return Err(format!(
                    "Electorate faction {n}: candidate_center has length {}, but dimensions = {}",
                    center.len(),
                    self.dimensions
                ));
            }
            if faction.popularity < 0.0 {
                return Err(format!(
                    "Electorate faction {n}: popularity must be non-negative"
                ));
            }
        }
        if self.factions.iter().map(|f| f.popularity).sum::<f64>() <= 0.0 {
            return Err("Electorate needs positive total popularity".to_string());
        }
        Ok(())
    }

    pub fn new_sim(&self, sim: &Sim) -> ElectorateSim {
        // `Config::validate` reports these as errors; this guards direct callers.
        if let Err(e) = self.validate() {
            panic!("{e}");
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

        ElectorateSim {
            dims: self.dimensions,
            distance_scaling: self.distance_function.clone(),
            cand_distribution: self.candidate_distribution,
            factions: self.factions.clone(),
            popularity_cdf,
            total_popularity: ttl_popularity,
            cand_positions: Array2::zeros((sim.ncand, self.dimensions)),
            cand_factions: vec![0; sim.ncand],
            in_group_like: vec![0.0f64; sim.ncand],
            faction_voters: vec![0; self.factions.len()],
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
            self.cand_distribution
                .draw(rng, center, spread, self.cand_positions.row_mut(icand));
            self.in_group_like[icand] = draw_ceiling(rng, faction.in_group_likability_ceiling);
        }
        log::debug!("candidate faction numbers: {:?}", self.cand_factions);
        log::debug!("faction candidate positions:\n{}", self.cand_positions);

        // Voters: each is assigned a faction weighted by popularity, drawn around
        // that faction's center, then scored against every candidate.
        self.faction_voters.fill(0);
        for mut vtr_scores in scores.rows_mut() {
            let r = rng.random_range(0.0..self.total_popularity);
            let vfac = self.popularity_cdf.partition_point(|&c| c <= r);
            self.faction_voters[vfac] += 1;
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
            faction_voters: self.faction_voters.clone(),
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
    fn candidate_distribution_defaults_to_gaussian_and_accepts_lowercase() {
        let parse = |extra: &str| -> Electorate {
            serde_json::from_str(&format!(
                r#"{{"dimensions": 1, {extra} "factions": [{{"popularity": 1.0, "voter_center": [0.0], "voter_spread": 1.0}}]}}"#
            ))
            .unwrap()
        };
        assert_eq!(
            parse("").candidate_distribution,
            PositionDistribution::Gaussian
        );
        for name in ["Uniform", "uniform"] {
            assert_eq!(
                parse(&format!(r#""candidate_distribution": "{name}","#)).candidate_distribution,
                PositionDistribution::Uniform
            );
        }
    }

    /// Uniform positions stay within `spread` of the center and fill the ball
    /// evenly: half its volume lies within `spread * (1/2)^(1/D)`, so about
    /// half the points should too.
    #[test]
    fn uniform_positions_fill_the_ball_around_the_center() {
        let mut rng = StdRng::seed_from_u64(6);
        let n = 20_000;
        for center in [vec![0.5], vec![1.0, -2.0], vec![0.0, 3.0, -1.0]] {
            let dims = center.len();
            let spread = 0.5;
            let half_volume_radius = spread * 0.5f64.powf(1.0 / dims as f64);
            let mut pos = ndarray::Array1::zeros(dims);
            let mut inner = 0;
            let mut sum = vec![0.0; dims];
            for _ in 0..n {
                PositionDistribution::Uniform.draw(&mut rng, &center, spread, pos.view_mut());
                let dist = pos
                    .iter()
                    .zip(&center)
                    .map(|(p, c)| (p - c).powi(2))
                    .sum::<f64>()
                    .sqrt();
                assert!(dist <= spread, "{dims}-D: {pos} is {dist} from {center:?}");
                inner += (dist <= half_volume_radius) as usize;
                for (s, p) in sum.iter_mut().zip(&pos) {
                    *s += p;
                }
            }
            let inner_share = inner as f64 / n as f64;
            assert!((inner_share - 0.5).abs() < 0.02, "{dims}-D: {inner_share}");
            for (s, c) in sum.iter().zip(&center) {
                assert_abs_diff_eq!(s / n as f64, *c, epsilon = 0.01);
            }
        }
    }

    /// The default draws exactly what it always has -- one standard normal per
    /// coordinate, in order -- so existing configs give the same elections.
    #[test]
    fn gaussian_positions_draw_what_they_always_have() {
        let center = [1.0, -2.0, 0.5];
        let spread = 0.7;
        let mut pos = ndarray::Array1::zeros(3);
        let mut rng = StdRng::seed_from_u64(9);
        let mut by_hand = StdRng::seed_from_u64(9);
        for _ in 0..100 {
            PositionDistribution::Gaussian.draw(&mut rng, &center, spread, pos.view_mut());
            for (p, &c) in pos.iter().zip(&center) {
                let z: f64 = by_hand.sample(StandardNormal);
                assert_eq!(*p, c + z * spread);
            }
        }
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

    fn electorate(dimensions: usize, factions: Vec<Faction>) -> Electorate {
        Electorate {
            dimensions,
            distance_function: DistanceFunction::NegativeEuclidean,
            candidate_distribution: PositionDistribution::Gaussian,
            factions,
        }
    }

    #[test]
    fn validate_accepts_a_well_formed_electorate() {
        let e = electorate(2, vec![faction([0.0, 0.0], 1.0), faction([1.0, 1.0], 0.0)]);
        assert!(e.validate().is_ok());
    }

    #[test]
    fn validate_rejects_zero_dimensions() {
        let e = electorate(0, vec![]);
        assert_eq!(e.validate().unwrap_err(), "Electorate needs dimensions > 0");
    }

    #[test]
    fn validate_rejects_no_factions() {
        let e = electorate(2, vec![]);
        assert_eq!(
            e.validate().unwrap_err(),
            "Electorate needs at least one faction"
        );
    }

    #[test]
    fn validate_rejects_mismatched_center_lengths() {
        // 2 coords, dimensions = 3
        let e = electorate(3, vec![faction([0.0, 0.0], 1.0)]);
        assert_eq!(
            e.validate().unwrap_err(),
            "Electorate faction 1: voter_center has length 2, but dimensions = 3"
        );

        let mut f = faction([0.0, 0.0], 1.0);
        f.candidate_center = Some(vec![0.0]);
        let e = electorate(2, vec![faction([0.0, 0.0], 1.0), f]);
        assert_eq!(
            e.validate().unwrap_err(),
            "Electorate faction 2: candidate_center has length 1, but dimensions = 2"
        );
    }

    #[test]
    fn validate_rejects_negative_popularity() {
        let e = electorate(2, vec![faction([0.0, 0.0], 2.0), faction([1.0, 1.0], -1.0)]);
        assert_eq!(
            e.validate().unwrap_err(),
            "Electorate faction 2: popularity must be non-negative"
        );
    }

    #[test]
    fn validate_rejects_zero_total_popularity() {
        let e = electorate(2, vec![faction([0.0, 0.0], 0.0), faction([1.0, 1.0], 0.0)]);
        assert_eq!(
            e.validate().unwrap_err(),
            "Electorate needs positive total popularity"
        );
    }

    /// Every voter is counted in exactly one faction, in proportion to
    /// popularity: none in a faction with popularity 0.
    #[test]
    fn faction_voters_counts_each_factions_voters() {
        let e = electorate(
            2,
            vec![
                faction([0.0, 0.0], 1.0),
                faction([1.0, 1.0], 0.0),
                faction([2.0, 2.0], 3.0),
            ],
        );
        let sim = Sim::new(3, 400);
        let mut esim = e.new_sim(&sim);
        let mut scores = Array2::zeros((400, 3));
        let mut rng = StdRng::seed_from_u64(5);
        for _ in 0..2 {
            esim.add_to_scores(&mut scores, &mut rng);
            let counts = &esim.make_faction_info(vec![], &[]).faction_voters;
            assert_eq!(counts.iter().sum::<u32>(), 400, "{counts:?}");
            assert_eq!(counts[1], 0);
            assert!(counts[2] > 2 * counts[0], "{counts:?}");
        }
    }

    #[test]
    #[should_panic(expected = "voter_center has length 2, but dimensions = 3")]
    fn new_sim_panics_on_an_invalid_electorate() {
        electorate(3, vec![faction([0.0, 0.0], 1.0)]).new_sim(&Sim::new(3, 5));
    }

    #[test]
    fn scores_are_negative_distance_when_spreads_and_likability_are_zero() {
        // One faction, voters pinned to the center (spread 0) but candidates
        // scattered. Every voter should score candidate c at -||candidate_pos[c]||.
        let params = Electorate {
            dimensions: 2,
            distance_function: DistanceFunction::NegativeEuclidean,
            candidate_distribution: PositionDistribution::Gaussian,
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
            candidate_distribution: PositionDistribution::Gaussian,
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
            candidate_distribution: PositionDistribution::Gaussian,
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
