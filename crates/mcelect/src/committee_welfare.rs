// © Copyright 2026 Topher Cawlfield
// SPDX-License-Identifier: Apache-2.0

//! How much the voters value an elected committee, under several welfare
//! functions, and how that compares with the best and the average committee.
//!
//! Each voter's utilities are first rescaled over the whole candidate field to
//! [0, 1]. A voter's value for a committee is then a weighted sum of their
//! rescaled utilities for its members, best member first (an ordered weighted
//! average); the welfare of the committee, `W(S)`, is the mean of that over
//! voters. The weights decide what kind of representation is rewarded:
//!
//! * [`Welfare::Additive`] -- equal weights. Majoritarian: the best committee
//!   is the k individually best-liked candidates.
//! * [`Welfare::Harmonic`] -- weights 1, 1/2, 1/3, ... as in Proportional
//!   Approval Voting. Proportional.
//! * [`Welfare::ChamberlinCourant`] -- only a voter's best representative
//!   counts.
//!
//! Regret is `(W* - W(S)) / (W* - W̄)`, with `W*` the best and `W̄` the mean
//! welfare over every possible committee: 0 for the best committee, 1 for an
//! average one, as single-winner regret is. Both are found by searching every
//! committee, which is by far the costliest part (see [`search`]).
//!
//! The working buffers are allocated once, in [`WelfareEval::new`] and
//! [`SearchBufs::new`], and reused for every trial. The free functions take
//! their buffers as arguments so the benchmarks can compare that with
//! allocating fresh ones.

use ndarray::{Array2, ArrayView2};

use crate::sim::Sim;

/// A welfare function over committees: how a voter's utilities for the
/// members, sorted best first, are weighted. Configured in a multi-winner
/// config's `committee_welfare` list:
///
/// ```toml
/// committee_welfare = ["Additive", "Harmonic", "ChamberlinCourant"]
/// [[committee_welfare]]
/// Owa = { weights = [1.0, 0.5, 0.25], colname = "geo" }
/// ```
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub enum Welfare {
    /// Weights 1, 1, 1, ...: every member counts equally.
    Additive,
    /// Weights 1, 1/2, 1/3, ...: as in Proportional Approval Voting.
    Harmonic,
    /// Weights 1, 0, 0, ...: only the voter's best representative counts.
    ChamberlinCourant,
    /// Explicit weights, best member first. Padded with zeros to the
    /// committee size.
    Owa { weights: Vec<f64>, colname: String },
}

impl Welfare {
    /// Check that an `Owa`'s weights make sense for a committee of `k`.
    pub fn validate(&self, k: usize) -> Result<(), String> {
        let Welfare::Owa { weights, colname } = self else {
            return Ok(());
        };
        if colname.is_empty() {
            return Err("Owa needs a non-empty `colname`".to_string());
        }
        if weights.len() > k {
            return Err(format!(
                "Owa has {} weights, more than committee_size = {k}",
                weights.len()
            ));
        }
        if weights.iter().any(|w| !(w.is_finite() && *w >= 0.0)) {
            return Err("Owa weights must be finite and non-negative".to_string());
        }
        if weights.iter().sum::<f64>() <= 0.0 {
            return Err("Owa needs at least one positive weight".to_string());
        }
        Ok(())
    }

    /// The name this welfare function's results go under.
    pub fn colname(&self) -> String {
        match self {
            Welfare::Additive => "add".to_string(),
            Welfare::Harmonic => "pav".to_string(),
            Welfare::ChamberlinCourant => "cc".to_string(),
            Welfare::Owa { colname, .. } => colname.clone(),
        }
    }

    /// The weights for a committee of `k`, scaled to sum to 1 so that welfare
    /// stays in [0, 1].
    pub fn weights(&self, k: usize) -> Vec<f64> {
        let mut w: Vec<f64> = match self {
            Welfare::Additive => vec![1.0; k],
            Welfare::Harmonic => (1..=k).map(|j| 1.0 / j as f64).collect(),
            Welfare::ChamberlinCourant => (0..k).map(|j| if j == 0 { 1.0 } else { 0.0 }).collect(),
            Welfare::Owa { weights, .. } => {
                assert!(
                    weights.len() <= k,
                    "Owa has {} weights for a committee of {k}",
                    weights.len()
                );
                let mut w = weights.clone();
                w.resize(k, 0.0);
                w
            }
        };
        let total: f64 = w.iter().sum();
        assert!(total > 0.0, "welfare weights must have a positive sum");
        for x in w.iter_mut() {
            *x /= total;
        }
        w
    }
}

/// Committee welfare for one trial runner: the welfare functions' weights and
/// every buffer needed to evaluate them, allocated once.
pub struct WelfareEval {
    /// The committee size
    k: usize,
    /// Each welfare function's [`Welfare::colname`], in configured order.
    colnames: Vec<String>,
    /// One row per welfare function, `k` weights each.
    weights: Array2<f64>,
    /// Rescaled utilities, transposed from `Sim::scores`: `ncand` x `nvtr`, so
    /// one candidate's utilities for every voter are contiguous.
    norm_t: Array2<f64>,
    search: SearchBufs,
    /// One voter's utilities for a committee's members (`k`).
    scratch: Vec<f64>,
    /// For a committee: the sum over voters of each voter's j-th best member's
    /// utility (`k`).
    col_sums: Vec<f64>,
    /// `W*` this trial, per welfare function.
    pub best: Vec<f64>,
    /// `W̄` this trial, per welfare function.
    pub mean: Vec<f64>,
}

impl WelfareEval {
    pub fn new(welfare: &[Welfare], sim: &Sim, k: usize) -> WelfareEval {
        assert!(
            (1..=sim.ncand).contains(&k),
            "committee size {k} with {} candidates",
            sim.ncand
        );
        let mut weights = Array2::zeros((welfare.len(), k));
        for (mut row, w) in weights.rows_mut().into_iter().zip(welfare) {
            row.assign(&ndarray::Array1::from(w.weights(k)));
        }
        WelfareEval {
            k,
            colnames: welfare.iter().map(Welfare::colname).collect(),
            weights,
            norm_t: Array2::zeros((sim.ncand, sim.nvtr)),
            search: SearchBufs::new(sim.nvtr, k),
            scratch: vec![0.0; k],
            col_sums: vec![0.0; k],
            best: vec![0.0; welfare.len()],
            mean: vec![0.0; welfare.len()],
        }
    }

    /// Each welfare function's name, in the order of every per-function
    /// slice here ([`best`](Self::best), [`mean`](Self::mean), `score`'s output).
    pub fn colnames(&self) -> &[String] {
        &self.colnames
    }

    /// Once per trial, after the election: rescale the utilities and search
    /// every committee for [`best`](Self::best) and [`mean`](Self::mean).
    pub fn prepare(&mut self, sim: &Sim) {
        normalize_into(sim.scores.view(), &mut self.norm_t);
        search(
            self.norm_t.view(),
            self.weights.view(),
            &mut self.search,
            &mut self.best,
            &mut self.mean,
        );
    }

    /// `W(committee)` for every welfare function, into `out`. Needs
    /// [`prepare`](Self::prepare) first.
    pub fn score(&mut self, committee: &[usize], out: &mut [f64]) {
        assert_eq!(committee.len(), self.k, "committee of the wrong size");
        self.col_sums.fill(0.0);
        for ivtr in 0..self.norm_t.ncols() {
            for (u, &icand) in self.scratch.iter_mut().zip(committee) {
                *u = self.norm_t[(icand, ivtr)];
            }
            self.scratch.sort_unstable_by(|a, b| b.total_cmp(a));
            for (sum, u) in self.col_sums.iter_mut().zip(&self.scratch) {
                *sum += u;
            }
        }
        welfare_from_col_sums(
            self.weights.view(),
            &self.col_sums,
            self.norm_t.ncols(),
            out,
        );
    }

    /// The regret for welfare function `i` of a committee whose welfare is
    /// `value`. 0 when every committee is equally good.
    pub fn regret(&self, i: usize, value: f64) -> f64 {
        let span = self.best[i] - self.mean[i];
        if span > 0.0 {
            (self.best[i] - value) / span
        } else {
            0.0
        }
    }
}

/// Rescale each voter's utilities (a row of `scores`, `nvtr` x `ncand`) to
/// [0, 1] over the whole candidate field, writing them transposed into
/// `norm_t` (`ncand` x `nvtr`). A voter indifferent between every candidate
/// gets all zeros: they don't care who's elected.
pub fn normalize_into(scores: ArrayView2<f64>, norm_t: &mut Array2<f64>) {
    assert_eq!(norm_t.dim(), (scores.ncols(), scores.nrows()));
    for (ivtr, row) in scores.rows().into_iter().enumerate() {
        let lo = row.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let span = hi - lo;
        for (icand, &u) in row.iter().enumerate() {
            norm_t[(icand, ivtr)] = if span > 0.0 { (u - lo) / span } else { 0.0 };
        }
    }
}

/// Working memory for [`search`]: for each depth `d` of the search, every
/// voter's utilities for the first `d + 1` members chosen so far, sorted best
/// first. Laid out flat, voter-major, `k` slots per voter.
pub struct SearchBufs {
    k: usize,
    levels: Vec<Vec<f64>>,
    col_sums: Vec<f64>,
    /// One committee's welfare, per welfare function. Sized on first use.
    welfare: Vec<f64>,
}

impl SearchBufs {
    pub fn new(nvtr: usize, k: usize) -> SearchBufs {
        SearchBufs {
            k,
            // The last member is folded straight into `col_sums`, so the
            // deepest level is never stored.
            levels: (0..k.saturating_sub(1))
                .map(|_| vec![0.0; nvtr * k])
                .collect(),
            col_sums: vec![0.0; k],
            welfare: Vec::new(),
        }
    }
}

/// Find the best (`best`) and mean (`mean`) welfare over every committee of
/// `bufs`' size, for each row of `weights`, by visiting each committee.
///
/// A depth-first walk over the combinations: each level keeps every voter's
/// chosen members' utilities sorted, and each committee costs one pass over
/// the voters, about `nvtr * k` work, however many welfare functions there
/// are. With `m` candidates the whole search is about `nvtr * sum(d * C(m, d))`
/// for `d` up to `k`: dozens of times a trial's other work at 12 choose 5.
pub fn search(
    norm_t: ArrayView2<f64>,
    weights: ArrayView2<f64>,
    bufs: &mut SearchBufs,
    best: &mut [f64],
    mean: &mut [f64],
) {
    let (ncand, nvtr) = norm_t.dim();
    let k = bufs.k;
    assert!(norm_t.is_standard_layout(), "norm_t must be row-major");
    assert!((1..=ncand).contains(&k));
    assert_eq!(weights.dim(), (best.len(), k));
    assert_eq!(bufs.levels.first().map_or(nvtr * k, Vec::len), nvtr * k);

    best.fill(f64::NEG_INFINITY);
    mean.fill(0.0); // a running sum until the end
    bufs.welfare.resize(best.len(), 0.0);
    let mut walk = Walk {
        norm_t,
        weights,
        nvtr,
        k,
        bufs,
        best,
        mean,
        count: 0,
    };
    walk.descend(0, 0);
    let count = walk.count as f64;
    for m in mean.iter_mut() {
        *m /= count; // from a running sum to the mean
    }
}

/// The state of one [`search`], so the recursion needn't pass it all along.
struct Walk<'a> {
    norm_t: ArrayView2<'a, f64>,
    weights: ArrayView2<'a, f64>,
    nvtr: usize,
    k: usize,
    bufs: &'a mut SearchBufs,
    best: &'a mut [f64],
    mean: &'a mut [f64],
    count: usize,
}

impl Walk<'_> {
    /// `depth` members are chosen (sorted per voter in `levels[depth - 1]`);
    /// try each candidate from `start` on as the next.
    fn descend(&mut self, depth: usize, start: usize) {
        let ncand = self.norm_t.nrows();
        let k = self.k;
        if depth == k - 1 {
            for icand in start..ncand {
                self.finish_committee(depth, icand);
            }
            return;
        }
        // Leave enough candidates after this one to fill the committee.
        for icand in start..=ncand - (k - depth) {
            let utils = self.norm_t.row(icand);
            let utils = utils.as_slice().expect("row-major norm_t");
            let (done, rest) = self.bufs.levels.split_at_mut(depth);
            let next = &mut rest[0];
            for ivtr in 0..self.nvtr {
                let row = &mut next[ivtr * k..ivtr * k + depth + 1];
                let prev = match done.last() {
                    Some(prev) => &prev[ivtr * k..ivtr * k + depth],
                    None => &[][..],
                };
                insert_sorted(prev, utils[ivtr], row);
            }
            self.descend(depth + 1, icand + 1);
        }
    }

    /// Complete a committee with `icand`, folding each voter's utilities
    /// straight into the column sums, and record its welfare.
    fn finish_committee(&mut self, depth: usize, icand: usize) {
        let k = self.k;
        let utils = self.norm_t.row(icand);
        let utils = utils.as_slice().expect("row-major norm_t");
        let SearchBufs {
            levels,
            col_sums,
            welfare,
            ..
        } = &mut *self.bufs;
        col_sums.fill(0.0);
        for ivtr in 0..self.nvtr {
            let prev = match levels.last() {
                Some(prev) => &prev[ivtr * k..ivtr * k + depth],
                None => &[][..],
            };
            let u = utils[ivtr];
            // Merge `u` into the sorted `prev`, adding as we go.
            let mut j = 0;
            let mut placed = false;
            for &x in prev {
                if !placed && u > x {
                    col_sums[j] += u;
                    j += 1;
                    placed = true;
                }
                col_sums[j] += x;
                j += 1;
            }
            if !placed {
                col_sums[j] += u;
            }
        }
        welfare_from_col_sums(self.weights, col_sums, self.nvtr, welfare);
        for ((b, m), &w) in self
            .best
            .iter_mut()
            .zip(self.mean.iter_mut())
            .zip(&*welfare)
        {
            *b = b.max(w);
            *m += w;
        }
        self.count += 1;
    }
}

/// Write `prev` (sorted best first) with `u` inserted in order into `out`,
/// which is one longer.
fn insert_sorted(prev: &[f64], u: f64, out: &mut [f64]) {
    let pos = prev.iter().position(|&x| u > x).unwrap_or(prev.len());
    out[..pos].copy_from_slice(&prev[..pos]);
    out[pos] = u;
    out[pos + 1..].copy_from_slice(&prev[pos..]);
}

/// Mean welfare over voters for each row of `weights`, from the column sums:
/// `sum over voters, j of w[j] * u(j-th best)` is `sum over j of w[j] * col_sums[j]`.
fn welfare_from_col_sums(weights: ArrayView2<f64>, col_sums: &[f64], nvtr: usize, out: &mut [f64]) {
    for (o, w) in out.iter_mut().zip(weights.rows()) {
        *o = w.iter().zip(col_sums).map(|(w, s)| w * s).sum::<f64>() / nvtr as f64;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;
    use rand::RngExt as _;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    const ALL: [Welfare; 3] = [
        Welfare::Additive,
        Welfare::Harmonic,
        Welfare::ChamberlinCourant,
    ];

    fn eval_for(scores: Array2<f64>, welfare: &[Welfare], k: usize) -> WelfareEval {
        let (nvtr, ncand) = scores.dim();
        let mut sim = Sim::new(ncand, nvtr);
        sim.scores = scores;
        let mut eval = WelfareEval::new(welfare, &sim, k);
        eval.prepare(&sim);
        eval
    }

    fn score(eval: &mut WelfareEval, committee: &[usize]) -> Vec<f64> {
        let mut out = vec![0.0; eval.best.len()];
        eval.score(committee, &mut out);
        out
    }

    /// Every k-subset of 0..m, the slow obvious way.
    fn combinations(m: usize, k: usize) -> Vec<Vec<usize>> {
        if k == 0 {
            return vec![vec![]];
        }
        (k - 1..m)
            .flat_map(|last| {
                combinations(last, k - 1).into_iter().map(move |mut c| {
                    c.push(last);
                    c
                })
            })
            .collect()
    }

    #[test]
    fn weights_are_scaled_to_sum_to_1() {
        assert_eq!(Welfare::Additive.weights(4), [0.25; 4]);
        assert_eq!(Welfare::ChamberlinCourant.weights(3), [1.0, 0.0, 0.0]);
        let h = Welfare::Harmonic.weights(3); // 1, 1/2, 1/3 over 11/6
        assert_abs_diff_eq!(
            h.as_slice(),
            [6.0 / 11.0, 3.0 / 11.0, 2.0 / 11.0].as_slice()
        );
        let owa = Welfare::Owa {
            weights: vec![2.0, 1.0],
            colname: "x".to_string(),
        };
        assert_abs_diff_eq!(
            owa.weights(4).as_slice(),
            [2.0 / 3.0, 1.0 / 3.0, 0.0, 0.0].as_slice()
        );
    }

    #[test]
    fn owa_weights_are_validated_against_the_committee_size() {
        let owa = |weights: Vec<f64>, colname: &str| Welfare::Owa {
            weights,
            colname: colname.to_string(),
        };
        assert!(owa(vec![1.0, 0.5], "geo").validate(2).is_ok());
        assert!(Welfare::Harmonic.validate(2).is_ok());
        let cases = [
            (owa(vec![1.0], ""), "Owa needs a non-empty `colname`"),
            (
                owa(vec![1.0, 0.5, 0.25], "geo"),
                "Owa has 3 weights, more than committee_size = 2",
            ),
            (
                owa(vec![1.0, -0.5], "geo"),
                "Owa weights must be finite and non-negative",
            ),
            (
                owa(vec![f64::NAN], "geo"),
                "Owa weights must be finite and non-negative",
            ),
            (
                owa(vec![0.0, 0.0], "geo"),
                "Owa needs at least one positive weight",
            ),
            (owa(vec![], "geo"), "Owa needs at least one positive weight"),
        ];
        for (welfare, err) in cases {
            assert_eq!(welfare.validate(2).unwrap_err(), err, "{welfare:?}");
        }
    }

    #[test]
    fn normalize_rescales_each_voter_and_zeroes_the_indifferent() {
        let scores = array![[-4.0, 0.0, -2.0], [2.0, 2.0, 2.0]];
        let mut norm_t = Array2::zeros((3, 2));
        normalize_into(scores.view(), &mut norm_t);
        assert_eq!(norm_t, array![[0.0, 0.0], [1.0, 0.0], [0.5, 0.0]]);
    }

    /// Worked by hand. Rescaled, voter A is [0, 0.25, 0.5, 1] and voter B
    /// [1, 0, 0, 0].
    #[test]
    fn hand_computed_welfare() {
        let scores = array![[0.0, 1.0, 2.0, 4.0], [3.0, 0.0, 0.0, 0.0]];
        let mut eval = eval_for(scores, &ALL, 2);

        // Both voters' sorted utilities are [1, 0]: column sums [2, 0].
        let w = score(&mut eval, &[0, 3]);
        assert_abs_diff_eq!(
            w.as_slice(),
            [0.5, 2.0 / 3.0, 1.0].as_slice(),
            epsilon = 1e-12
        );

        // A's are [0.5, 0.25], B's [0, 0]: column sums [0.5, 0.25].
        let w = score(&mut eval, &[1, 2]);
        let pav = (0.5 * 2.0 / 3.0 + 0.25 / 3.0) / 2.0;
        assert_abs_diff_eq!(
            w.as_slice(),
            [0.1875, pav, 0.25].as_slice(),
            epsilon = 1e-12
        );
    }

    /// `prepare`'s best and mean match scoring every committee one by one,
    /// across committee sizes from 1 to the whole field.
    #[test]
    fn search_matches_scoring_every_committee() {
        let mut rng = StdRng::seed_from_u64(7);
        let (nvtr, ncand) = (9, 7);
        for k in 1..=ncand {
            let scores = Array2::from_shape_fn((nvtr, ncand), |_| rng.random::<f64>() - 0.5);
            let mut eval = eval_for(scores, &ALL, k);
            let committees = combinations(ncand, k);
            let mut best = vec![f64::NEG_INFINITY; ALL.len()];
            let mut mean = vec![0.0; ALL.len()];
            for c in &committees {
                for (i, w) in score(&mut eval, c).into_iter().enumerate() {
                    best[i] = best[i].max(w);
                    mean[i] += w / committees.len() as f64;
                }
            }
            assert_abs_diff_eq!(eval.best.as_slice(), best.as_slice(), epsilon = 1e-12);
            assert_abs_diff_eq!(eval.mean.as_slice(), mean.as_slice(), epsilon = 1e-12);
        }
    }

    /// Regret is 0 for the best committee and averages 1 over all of them.
    #[test]
    fn regret_is_0_at_best_and_1_on_average() {
        let mut rng = StdRng::seed_from_u64(11);
        let (nvtr, ncand, k) = (15, 6, 3);
        let scores = Array2::from_shape_fn((nvtr, ncand), |_| rng.random::<f64>());
        let mut eval = eval_for(scores, &ALL, k);
        let committees = combinations(ncand, k);
        for i in 0..ALL.len() {
            let regrets: Vec<f64> = committees
                .iter()
                .map(|c| {
                    let w = score(&mut eval, c)[i];
                    eval.regret(i, w)
                })
                .collect();
            let lowest = regrets.iter().copied().fold(f64::INFINITY, f64::min);
            assert_abs_diff_eq!(lowest, 0.0, epsilon = 1e-12);
            let avg = regrets.iter().sum::<f64>() / regrets.len() as f64;
            assert_abs_diff_eq!(avg, 1.0, epsilon = 1e-12);
        }
    }

    /// When all committees are equally good there's nothing to regret.
    #[test]
    fn regret_is_0_when_every_committee_ties() {
        let mut eval = eval_for(Array2::from_elem((4, 5), 1.0), &ALL, 2);
        let w = score(&mut eval, &[0, 1])[0];
        assert_eq!(eval.regret(0, w), 0.0);
    }

    /// The point of the exercise: two polarized blocs of 6 and 3 voters, 3
    /// seats. Additive welfare is best served by handing the majority every
    /// seat; Harmonic and Chamberlin-Courant by giving the minority one.
    #[test]
    fn harmonic_and_cc_reward_representing_the_minority() {
        let majority = [1.0, 0.95, 0.9, 0.0, 0.0, 0.0];
        let minority = [0.0, 0.0, 0.0, 1.0, 0.95, 0.9];
        let rows: Vec<[f64; 6]> = [[majority; 6].as_slice(), [minority; 3].as_slice()].concat();
        let scores = Array2::from_shape_fn((9, 6), |(v, c)| rows[v][c]);
        let mut eval = eval_for(scores, &ALL, 3);
        let (add, pav, cc) = (0, 1, 2);

        let sweep = score(&mut eval, &[0, 1, 2]); // the majority takes all
        let split = score(&mut eval, &[0, 1, 3]); // 2 seats to 1

        assert_abs_diff_eq!(sweep[add], eval.best[add], epsilon = 1e-12);
        assert!(split[add] < sweep[add]);

        assert_abs_diff_eq!(split[pav], eval.best[pav], epsilon = 1e-12);
        assert!(sweep[pav] < split[pav]);

        assert_abs_diff_eq!(split[cc], eval.best[cc], epsilon = 1e-12);
        assert!(sweep[cc] < split[cc]);
    }

    #[test]
    fn insert_sorted_places_the_new_value_in_order() {
        let mut out = [0.0; 4];
        insert_sorted(&[0.9, 0.5, 0.1], 0.6, &mut out);
        assert_eq!(out, [0.9, 0.6, 0.5, 0.1]);
        insert_sorted(&[0.9, 0.5, 0.1], 1.0, &mut out);
        assert_eq!(out, [1.0, 0.9, 0.5, 0.1]);
        insert_sorted(&[0.9, 0.5, 0.1], 0.0, &mut out);
        assert_eq!(out, [0.9, 0.5, 0.1, 0.0]);
        let mut one = [0.0];
        insert_sorted(&[], 0.3, &mut one);
        assert_eq!(one, [0.3]);
    }
}
