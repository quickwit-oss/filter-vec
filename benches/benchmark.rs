use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use rand::{Rng, distributions::Uniform};

fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("filter-interval");
    let mut rng= rand::thread_rng();
    let input: Vec<u32> = (&mut rng).sample_iter(Uniform::from(0..16)).take(1 << 20).collect();
    let mut output = Vec::with_capacity(input.len());
    group.throughput(Throughput::Elements(input.len() as u64));
    group.bench_function("avx2", |b| b.iter(|| filter_vec::avx2::filter_vec(&input, 4..=12, &mut output)));
    group.bench_function("avx512", |b| b.iter(|| filter_vec::avx512::filter_vec(&input, 4..=12, &mut output)));
    group.bench_function("scalar_iterator", |b| b.iter(|| filter_vec::filter_vec_iter(&input, 4..=12, &mut output)));
    group.bench_function("scalar_forloop", |b| b.iter(|| filter_vec::filter_vec_scalar(&input, 4..=12, &mut output)));
    group.bench_function("scalar_nobranch", |b| b.iter(|| filter_vec::filter_vec_nobranch(&input, 4..=12, &mut output)));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
