use rand::Rng;
use std::time::Instant;
use tnn_core::tensor::Tensor;

const M: usize = 2048;
const K: usize = 2048;
const N: usize = 2048;
const WARMUP_RUNS: usize = 1;
const MEASURE_RUNS: usize = 5;

fn random_tensor(shape: &[usize]) -> Tensor {
    let mut rng = rand::rng();
    let size: usize = shape.iter().product();
    let data: Vec<f32> = (0..size).map(|_| rng.random()).collect();
    Tensor::from(shape, data, false)
}
fn matmul_flops(m: usize, k: usize, n: usize) -> f64 {
    2.0 * m as f64 * k as f64 * n as f64
}

fn benchmark_matmul() {
    let a = random_tensor(&[M, K]);
    let b = random_tensor(&[K, N]);

    // Warm-up (important)
    for _ in 0..WARMUP_RUNS {
        let _ = a.matmul(b.clone());
    }

    let mut total_time = 0.0;

    for _ in 0..MEASURE_RUNS {
        let start = Instant::now();
        let _ = a.matmul(b.clone());
        let elapsed = start.elapsed().as_secs_f64();
        total_time += elapsed;
    }

    let avg_time = total_time / MEASURE_RUNS as f64;
    let flops = matmul_flops(M, K, N);
    let gflops = flops / avg_time / 1e9;

    println!("Matrix size: {}x{} * {}x{}", M, K, K, N);
    println!("Average time: {:.6} sec", avg_time);
    println!("Achieved performance: {:.2} GFLOPS", gflops);
}

fn main() {
    benchmark_matmul();

    //let t1 = Tensor::from(
    //    &[3, 3],
    //    [1., 2., 3., 4., 5., 6., 7., 8., 9.].to_vec(),
    //    false,
    //);
    //let t2 = Tensor::from(
    //    &[3, 3],
    //    [1., 2., 3., 4., 5., 6., 7., 8., 9.].to_vec(),
    //    false,
    //);
    //
    //println!("TENSOR 1 - {:?}", t1.data);
    //println!("TENSOR 2 - {:?}", t2.data);
    //
    //let t3 = t1.matmul(t2).unwrap();
    //println!("t3 - {:?}", t3.data);
}
