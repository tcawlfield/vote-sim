use ndarray::Array2;

/// CovMatrix computes the covariance matrix between candidates.
/// Each voter's utilities/scores are treated as random variates.
/// Note that correlation coefficients can be quickly derived from this:
/// cc[ix, iy] = elements[ix, iy] / (sqrt(elements[ix, ix] * elements[iy, iy]))
/// correlation coefficients may be more meaningful because the scale
/// of the utilities is arbitrary.
pub struct CovMatrix {
    pub elements: Array2<f64>,
    pub mean: Vec<f64>,
}

impl CovMatrix {
    pub fn new(ncand: usize) -> CovMatrix {
        CovMatrix {
            elements: Array2::zeros((ncand, ncand)),
            mean: vec![0.0; ncand],
        }
    }

    pub fn compute(&mut self, scores: &Array2<f64>) {
        let (ncit, ncand) = scores.dim();
        assert_eq!(self.elements.dim().0, ncand);
        self.mean.fill(0.0);
        self.elements.fill(0.0);
        for icit in 0..ncit {
            let n = (icit + 1) as f64;
            for ix in 0..ncand {
                let dx = scores[(icit, ix)] - self.mean[ix];
                self.mean[ix] += dx / n;
                for iy in 0..(ix + 1) {
                    self.elements[(ix, iy)] += dx * (scores[(icit, iy)] - self.mean[iy]);
                }
            }
        }
        for ix in 0..ncand {
            for iy in 0..(ix + 1) {
                self.elements[(ix, iy)] /= (ncit - 1) as f64;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use float_eq::assert_float_eq;
    use ndarray::array;

    #[test]
    fn test_cov_matrix() {
        // Using Python's numpy.cov for comparison.
        #[cfg_attr(rustfmt, rustfmt_skip)]
        {
            let utilities = array![
                [0.30900160, 2.24721985, 0.58539738, 2.85872826, 0.78623712],
                [2.09904798, 2.65914812, 1.22377722, 0.52028883, 0.66519608],
                [0.82317008, 1.86329343, 3.06071522, 1.38963001, 1.30919134],
                [3.13087807, 2.21673783, 1.3184892,  0.35922687, 2.73176796],
                [1.61064933, 0.84797068, 0.48842566, 0.16900879, 1.99823116],
                [2.01657254, 0.56396652, 0.79741721, 1.34662276, 3.12736477],
                [2.91759999, 0.44596168, 2.61924207, 0.17156912, 3.38542127],
                [1.55519431, 3.40639598, 3.07438678, 1.73254542, 1.83569564],
                [2.89235991, 0.65299382, 1.22978133, 1.48895542, 3.25960743],
                [2.97551052, 1.77379581, 3.12792367, 1.27941031, 1.54455224],
            ];
            let mut covm = CovMatrix::new(5);
            covm.compute(&utilities);
            let expected = array![
                [ 0.94020538, 0.0, 0.0, 0.0, 0.0],
                [-0.32073684,  1.01169877,  0.0,  0.0, 0.0],
                [ 0.16108757,  0.31368278,  1.19057637,  0.0, 0.0],
                [-0.49611513,  0.27153381, -0.00328159,  0.71388613, 0.0],
                [ 0.62033183, -0.70086438, -0.05275154, -0.31396175, 1.02640331]
            ];

            println!("Got covariance matrix: {}", covm.elements);
            for (computed, expect) in covm.elements.iter().zip(expected.iter()) {
                assert_float_eq!(*computed, *expect, abs <= 0.000001);
            }
        }

        /* From python:
        >>> import numpy as np
        >>> utilities = np.random.uniform(0., np.sqrt(12.), 50)
        >>> utilities = np.reshape(utilities, newshape=(5, 10))
        >>> print(str(utilities.transpose()))
        [[0.3090016  2.24721985 0.58539738 2.85872826 0.78623712]
         [2.09904798 2.65914812 1.22377722 0.52028883 0.66519608]
         [0.82317008 1.86329343 3.06071522 1.38963001 1.30919134]
         [3.13087807 2.21673783 1.3184892  0.35922687 2.73176796]
         [1.61064933 0.84797068 0.48842566 0.16900879 1.99823116]
         [2.01657254 0.56396652 0.79741721 1.34662276 3.12736477]
         [2.91759999 0.44596168 2.61924207 0.17156912 3.38542127]
         [1.55519431 3.40639598 3.07438678 1.73254542 1.83569564]
         [2.89235991 0.65299382 1.22978133 1.48895542 3.25960743]
         [2.97551052 1.77379581 3.12792367 1.27941031 1.54455224]]
        >>> np.cov(utilities)
        array([[ 0.94020538, -0.32073684,  0.16108757, -0.49611513,  0.62033183],
              [-0.32073684,  1.01169877,  0.31368278,  0.27153381, -0.70086438],
              [ 0.16108757,  0.31368278,  1.19057637, -0.00328159, -0.05275154],
              [-0.49611513,  0.27153381, -0.00328159,  0.71388613, -0.31396175],
              [ 0.62033183, -0.70086438, -0.05275154, -0.31396175,  1.02640331]])
        */
    }
}
