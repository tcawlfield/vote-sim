use ndarray::Array2;

pub struct CovMatrix {
    pub elements: Array2<f64>,
    mean: Vec<f64>,
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
