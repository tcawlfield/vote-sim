use criterion::{Criterion, criterion_group, criterion_main};
use mcelect::considerations::{Consideration, ConsiderationSim};
use mcelect::sim::Sim;

fn bench_issues_sim(c: &mut Criterion) {
    let issues: Consideration = serde_json::from_str(
        r#"{"Issues": [
        {
            "sigma": 1.4,
            "sigma_vtr": 1.6,
            "halfcsep": 1.2,
            "halfvsep": 1.1,
            "uniform": false
        },
        {
            "sigma": 2.4,
            "sigma_vtr": 2.6,
            "halfcsep": 1.2,
            "halfvsep": 1.1,
            "uniform": false
        },
        {
            "sigma": 3.4,
            "sigma_vtr": 3.6,
            "halfcsep": 1.2,
            "halfvsep": 1.1,
            "uniform": true
        }
    ]}"#,
    )
    .unwrap();
    let mut sim = Sim::new(12, 100_000);
    let mut issues_sim = issues.new_sim(&sim);
    let mut rng = rand::rng();
    c.bench_function("three_issues_12x100k", |b| {
        b.iter(|| issues_sim.add_to_scores(&mut sim.scores, &mut rng))
    });
}

criterion_group!(benches, bench_issues_sim);
criterion_main!(benches);
