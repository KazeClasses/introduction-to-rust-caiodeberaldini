use rand::{prelude::Distribution, rngs::StdRng, Rng, SeedableRng};
use statrs::distribution::MultivariateNormal;

pub struct State<const N_DIM: usize> {
    rng: StdRng,
    pub arr: [f64; N_DIM],
    proposal_distribution: MultivariateNormal,
}

fn log_likelihood<const N_DIM: usize>(arr: &[f64]) -> f64 {
    let mut p_x: f64 = 0.0;

    for i in 0..N_DIM{
        p_x += -(arr[i] * arr[i]) / 2.0;
    }
    return p_x;
}

impl<const N_DIM: usize> State<N_DIM> {
    pub fn new(seed: u64) -> Self {
        let rng = StdRng::seed_from_u64(seed);
        let arr = [0.0; N_DIM];
        let mean = vec![0.0; N_DIM];
        let mut cov = Vec::<f64>::new();
        for i in 0..N_DIM{
            for j in 0..N_DIM{
                if i == j{
                    cov.push(1.0);
                }
                else{
                    cov.push(0.0);
                }
            }
        }
        let proposal_distribution = MultivariateNormal::new(mean, cov).unwrap();
        State{rng, arr, proposal_distribution}
    }

    pub fn take_step(&mut self) {
        let binding = self.proposal_distribution.sample(&mut self.rng);
        let proposal= binding.as_slice();
        let ll = log_likelihood::<N_DIM>(proposal) - log_likelihood::<N_DIM>(&self.arr);
        // let u = Uniform::new(0.0, 1.0).unwrap().sample(&mut self.rng);
        let u = (self.rng.gen_range(0.0..1.0) as f64).ln();
        if u < ll{
            let mut new_loc: [f64; N_DIM] = [0.0; N_DIM];
            for i in 0..N_DIM{
                new_loc[i] = proposal[i]
            }
            self.arr = new_loc;
        }
    }
}
