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
//! average one, as single-winner regret is. `W̄` follows exactly from each
//! voter's ranking of the candidates. `W*` takes a search over committees,
//! which is by far the costliest part (see [`WelfareEval::prepare`]).
//!
//! [`WelfareEval`] allocates its working buffers once and reuses them for
//! every trial.

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
    /// One row per welfare function: the weight a voter's `i`-th favorite
    /// candidate gets on average over every committee (`ncand` each), so
    /// that `W̄` is a weighted sum over each voter's sorted utilities.
    rank_weights: Array2<f64>,
    /// The welfare functions whose `W*` is found by visiting every committee,
    /// by index, and their weights. Chamberlin-Courant isn't among them.
    exhaustive: Vec<usize>,
    exhaustive_weights: Array2<f64>,
    /// Chamberlin-Courant's index, and its branch-and-bound search.
    cc: Option<(usize, CcSearch)>,
    /// Rescaled utilities, transposed from `Sim::scores`: `ncand` x `nvtr`, so
    /// one candidate's utilities for every voter are contiguous.
    norm_t: Array2<f64>,
    /// For each depth `d` of the exhaustive search, every voter's utilities for
    /// the first `d + 1` members chosen so far, sorted best first. Flat,
    /// voter-major, `k` slots per voter. The last member is folded straight
    /// into `col_sums`, so there's no level for it.
    levels: Vec<Vec<f64>>,
    /// One voter's utilities for a committee's members (`k`), or for every
    /// candidate (`ncand`) when computing `W̄`.
    scratch: Vec<f64>,
    /// For a committee: the sum over voters of each voter's j-th best member's
    /// utility (`k`).
    col_sums: Vec<f64>,
    /// One committee's welfare, per exhaustively searched welfare function.
    welfare: Vec<f64>,
    /// `W*` this trial, per welfare function.
    pub best: Vec<f64>,
    /// `W̄` this trial, per welfare function.
    pub mean: Vec<f64>,
}

impl WelfareEval {
    pub fn new(welfare: &[Welfare], sim: &Sim, k: usize) -> WelfareEval {
        let (ncand, nvtr) = (sim.ncand, sim.nvtr);
        assert!(
            (1..=ncand).contains(&k),
            "committee size {k} with {ncand} candidates"
        );
        let weights = weight_rows(welfare.iter(), k);
        let cc = welfare
            .iter()
            .position(|w| *w == Welfare::ChamberlinCourant)
            .map(|i| (i, CcSearch::new(ncand, nvtr, k)));
        let exhaustive: Vec<usize> = (0..welfare.len())
            .filter(|&i| cc.as_ref().is_none_or(|(icc, _)| i != *icc))
            .collect();
        let levels = if exhaustive.is_empty() { 0 } else { k - 1 };
        WelfareEval {
            k,
            colnames: welfare.iter().map(Welfare::colname).collect(),
            rank_weights: rank_weights(weights.view(), ncand),
            exhaustive_weights: weight_rows(exhaustive.iter().map(|&i| &welfare[i]), k),
            welfare: vec![0.0; exhaustive.len()],
            exhaustive,
            weights,
            cc,
            norm_t: Array2::zeros((ncand, nvtr)),
            levels: (0..levels).map(|_| vec![0.0; nvtr * k]).collect(),
            scratch: vec![0.0; ncand.max(k)],
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

    /// Once per trial, after the election: rescale the utilities, then find
    /// [`mean`](Self::mean) and [`best`](Self::best) for every welfare function.
    ///
    /// The mean is cheap, one sort per voter. The best takes a search:
    /// Chamberlin-Courant's is a branch and bound that skips most committees,
    /// for about a tenth of the work. Every other function's visits them all,
    /// in a depth-first walk over the combinations that keeps every voter's
    /// chosen members' utilities sorted, about `nvtr * k` work per committee.
    /// That's about `nvtr * sum(d * C(m, d))` for `d` up to `k` with `m`
    /// candidates: dozens of times a trial's other work at 12 choose 5.
    pub fn prepare(&mut self, sim: &Sim) {
        normalize_into(sim.scores.view(), &mut self.norm_t);
        self.mean_welfare();
        if !self.exhaustive.is_empty() {
            for &i in &self.exhaustive {
                self.best[i] = f64::NEG_INFINITY;
            }
            self.descend(0, 0);
        }
        if let Some((i, cc)) = &mut self.cc {
            self.best[*i] = cc.best(self.norm_t.view());
        }
    }

    /// `W(committee)` for every welfare function, into `out`. Needs
    /// [`prepare`](Self::prepare) first.
    pub fn score(&mut self, committee: &[usize], out: &mut [f64]) {
        assert_eq!(committee.len(), self.k, "committee of the wrong size");
        let scratch = &mut self.scratch[..self.k];
        self.col_sums.fill(0.0);
        for ivtr in 0..self.norm_t.ncols() {
            for (u, &icand) in scratch.iter_mut().zip(committee) {
                *u = self.norm_t[(icand, ivtr)];
            }
            scratch.sort_unstable_by(|a, b| b.total_cmp(a));
            for (sum, u) in self.col_sums.iter_mut().zip(scratch.iter()) {
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

    /// `W̄` for every welfare function: sort each voter's utilities, best
    /// first, and weight them by [`rank_weights`](Self::rank_weights).
    fn mean_welfare(&mut self) {
        let (_, nvtr) = self.norm_t.dim();
        self.mean.fill(0.0);
        for utils in self.norm_t.columns() {
            for (u, &x) in self.scratch.iter_mut().zip(utils.iter()) {
                *u = x;
            }
            self.scratch.sort_unstable_by(|a, b| b.total_cmp(a));
            for (m, w) in self.mean.iter_mut().zip(self.rank_weights.rows()) {
                *m += w.iter().zip(&self.scratch).map(|(w, u)| w * u).sum::<f64>();
            }
        }
        for m in self.mean.iter_mut() {
            *m /= nvtr as f64;
        }
    }

    /// The exhaustive search: `depth` members are chosen (sorted per voter in
    /// `levels[depth - 1]`); try each candidate from `start` on as the next.
    fn descend(&mut self, depth: usize, start: usize) {
        let (ncand, nvtr) = self.norm_t.dim();
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
            let (done, rest) = self.levels.split_at_mut(depth);
            let next = &mut rest[0];
            for ivtr in 0..nvtr {
                let prev = match done.last() {
                    Some(prev) => &prev[ivtr * k..ivtr * k + depth],
                    None => &[][..],
                };
                insert_sorted(prev, utils[ivtr], &mut next[ivtr * k..ivtr * k + depth + 1]);
            }
            self.descend(depth + 1, icand + 1);
        }
    }

    /// Complete a committee with `icand`, folding each voter's utilities
    /// straight into the column sums, and keep its welfare if it's the best.
    fn finish_committee(&mut self, depth: usize, icand: usize) {
        let (_, nvtr) = self.norm_t.dim();
        let k = self.k;
        let utils = self.norm_t.row(icand);
        let utils = utils.as_slice().expect("row-major norm_t");
        self.col_sums.fill(0.0);
        for ivtr in 0..nvtr {
            let prev = match self.levels.last() {
                Some(prev) => &prev[ivtr * k..ivtr * k + depth],
                None => &[][..],
            };
            let u = utils[ivtr];
            // Merge `u` into the sorted `prev`, adding as we go.
            let mut j = 0;
            let mut placed = false;
            for &x in prev {
                if !placed && u > x {
                    self.col_sums[j] += u;
                    j += 1;
                    placed = true;
                }
                self.col_sums[j] += x;
                j += 1;
            }
            if !placed {
                self.col_sums[j] += u;
            }
        }
        welfare_from_col_sums(
            self.exhaustive_weights.view(),
            &self.col_sums,
            nvtr,
            &mut self.welfare,
        );
        for (&i, &w) in self.exhaustive.iter().zip(&self.welfare) {
            self.best[i] = self.best[i].max(w);
        }
    }
}

/// Branch and bound for the best Chamberlin-Courant committee, the one
/// maximizing the sum over voters of their best member's utility.
///
/// That welfare is submodular: a candidate adds less to a bigger committee.
/// So a committee `P` with `r` seats left to fill can't end up better than
/// `f(P)` plus the `r` largest gains any one remaining candidate would add to
/// `P` -- and a branch whose bound is no better than the best committee found
/// so far is skipped. The same gains are the children's values, so the bound
/// costs little beyond the plain search.
struct CcSearch {
    k: usize,
    /// For each depth `d`: each voter's best utility among the `d` members
    /// chosen so far (`nvtr` each; all 0 at depth 0).
    cur: Vec<Vec<f64>>,
    /// For each depth: each remaining candidate's gain over that committee,
    /// summed over voters (`ncand` each, from the depth's `start` on).
    gains: Vec<Vec<f64>>,
    /// The gains, sorted to find the largest (`ncand`).
    top: Vec<f64>,
    /// The best committee's welfare so far, summed over voters.
    best: f64,
}

impl CcSearch {
    fn new(ncand: usize, nvtr: usize, k: usize) -> CcSearch {
        CcSearch {
            k,
            cur: (0..k).map(|_| vec![0.0; nvtr]).collect(),
            gains: (0..k).map(|_| vec![0.0; ncand]).collect(),
            top: Vec::with_capacity(ncand),
            best: 0.0,
        }
    }

    /// The best committee's welfare, as a mean over voters.
    fn best(&mut self, norm_t: ArrayView2<f64>) -> f64 {
        self.best = f64::NEG_INFINITY;
        self.cur[0].fill(0.0);
        self.descend(norm_t, 0, 0, 0.0);
        self.best / norm_t.ncols() as f64
    }

    /// `depth` members are chosen, worth `value`; try each candidate from
    /// `start` on as the next, unless the bound rules all of them out.
    fn descend(&mut self, norm_t: ArrayView2<f64>, depth: usize, start: usize, value: f64) {
        let ncand = norm_t.nrows();
        let left = self.k - depth;
        let cur = &self.cur[depth];
        let gains = &mut self.gains[depth];
        for (gain, utils) in gains[start..]
            .iter_mut()
            .zip(norm_t.rows().into_iter().skip(start))
        {
            *gain = utils
                .iter()
                .zip(cur)
                .map(|(&u, &have)| (u - have).max(0.0))
                .sum();
        }
        self.top.clear();
        self.top.extend_from_slice(&gains[start..]);
        self.top.sort_unstable_by(|a, b| b.total_cmp(a));
        if value + self.top[..left].iter().sum::<f64>() <= self.best {
            return;
        }
        if left == 1 {
            for icand in start..ncand {
                self.best = self.best.max(value + self.gains[depth][icand]);
            }
            return;
        }
        // Leave enough candidates after this one to fill the committee.
        for icand in start..=ncand - left {
            let (done, rest) = self.cur.split_at_mut(depth + 1);
            for ((next, &have), &u) in rest[0].iter_mut().zip(&done[depth]).zip(norm_t.row(icand)) {
                *next = have.max(u);
            }
            let gain = self.gains[depth][icand];
            self.descend(norm_t, depth + 1, icand + 1, value + gain);
        }
    }
}

/// The welfare functions' weights for a committee of `k`, one row each.
fn weight_rows<'a>(welfare: impl Iterator<Item = &'a Welfare>, k: usize) -> Array2<f64> {
    let rows: Vec<Vec<f64>> = welfare.map(|w| w.weights(k)).collect();
    Array2::from_shape_fn((rows.len(), k), |(i, j)| rows[i][j])
}

/// For each row of `weights` (committee size `k`), the weight a voter's
/// `i`-th favorite of `ncand` candidates gets on average over every
/// committee. A committee's `j`-th best member is the voter's `i`-th favorite
/// (counting from 0) in `C(i, j) * C(ncand-1-i, k-1-j)` of the `C(ncand, k)`
/// committees: `j` of the `i` candidates they like better are in it, as are
/// `k-1-j` of those they like less.
fn rank_weights(weights: ArrayView2<f64>, ncand: usize) -> Array2<f64> {
    let k = weights.ncols();
    let committees = binomial(ncand, k);
    Array2::from_shape_fn((weights.nrows(), ncand), |(row, i)| {
        (0..k)
            .map(|j| weights[(row, j)] * binomial(i, j) * binomial(ncand - 1 - i, k - 1 - j))
            .sum::<f64>()
            / committees
    })
}

/// `C(n, r)`, as a float: exact for the sizes here, and never overflowing.
fn binomial(n: usize, r: usize) -> f64 {
    if r > n {
        return 0.0;
    }
    (0..r).fold(1.0, |c, i| c * (n - i) as f64 / (i + 1) as f64)
}

/// Rescale each voter's utilities (a row of `scores`, `nvtr` x `ncand`) to
/// [0, 1] over the whole candidate field, writing them transposed into
/// `norm_t` (`ncand` x `nvtr`). A voter indifferent between every candidate
/// gets all zeros: they don't care who's elected.
fn normalize_into(scores: ArrayView2<f64>, norm_t: &mut Array2<f64>) {
    for (ivtr, row) in scores.rows().into_iter().enumerate() {
        let lo = row.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let span = hi - lo;
        for (icand, &u) in row.iter().enumerate() {
            norm_t[(icand, ivtr)] = if span > 0.0 { (u - lo) / span } else { 0.0 };
        }
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

    #[test]
    fn rank_weights_spread_each_welfare_function_over_the_rankings() {
        let (ncand, k) = (6, 3);
        let rw = rank_weights(weight_rows(ALL.iter(), k).view(), ncand);
        for row in rw.rows() {
            assert_abs_diff_eq!(row.sum(), 1.0, epsilon = 1e-12);
        }
        // Additive: every candidate is in k / ncand of the committees, at
        // weight 1/k.
        for &w in rw.row(0) {
            assert_abs_diff_eq!(w, 1.0 / ncand as f64, epsilon = 1e-12);
        }
        // Chamberlin-Courant: the i-th favorite is a committee's best member
        // when the other k-1 members are all from the ncand-1-i liked less.
        for (i, &w) in rw.row(2).iter().enumerate() {
            let expected = binomial(ncand - 1 - i, k - 1) / binomial(ncand, k);
            assert_abs_diff_eq!(w, expected, epsilon = 1e-12);
        }
    }

    /// Voters in a few blocs, each with its own favorites plus individual
    /// noise: the structure branch and bound prunes on, unlike uniform noise.
    fn bloc_scores(rng: &mut StdRng, nvtr: usize, ncand: usize, nblocs: usize) -> Array2<f64> {
        let centers = Array2::from_shape_fn((nblocs, ncand), |_| rng.random::<f64>());
        Array2::from_shape_fn((nvtr, ncand), |(v, c)| {
            centers[(v % nblocs, c)] + 0.2 * rng.random::<f64>()
        })
    }

    /// Branch and bound finds the same best Chamberlin-Courant committee as
    /// trying them all, alone or alongside exhaustively searched functions.
    #[test]
    fn cc_branch_and_bound_finds_the_best_committee() {
        let mut rng = StdRng::seed_from_u64(3);
        let (nvtr, ncand) = (30, 10);
        for seed_round in 0..4 {
            for k in 1..=ncand {
                let scores = bloc_scores(&mut rng, nvtr, ncand, 2 + seed_round);
                for welfare in [&[Welfare::ChamberlinCourant][..], &ALL[..]] {
                    let icc = welfare.len() - 1;
                    let mut eval = eval_for(scores.clone(), welfare, k);
                    let naive = combinations(ncand, k)
                        .iter()
                        .map(|c| score(&mut eval, c)[icc])
                        .fold(f64::NEG_INFINITY, f64::max);
                    assert_abs_diff_eq!(eval.best[icc], naive, epsilon = 1e-12);
                }
            }
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
