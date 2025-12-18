use tnn_core::tensor::Tensor;

const M: usize = 512;
const K: usize = 512;
const N: usize = 512;

const WARMUP_RUNS: usize = 1;
const MEASURE_RUNS: usize = 5;
use rand::Rng;
use std::time::Instant;

fn random_tensor(shape: &[usize]) -> Tensor {
    let mut rng = rand::rng();
    let size: usize = shape.iter().product();
    let data: Vec<f32> = (0..size).map(|_| rng.random()).collect();
    Tensor::from(shape, data)
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

fn sd() {
    benchmark_matmul();
}

use ndarray::{Array, ArrayBase, Axis, Data, Dimension, Slice};
use num_traits::Zero;

/// Pad the edges of an array with zeros.
///
/// `pad_width` specifies the length of the padding at the beginning
/// and end of each axis.
///
/// **Panics** if `arr.ndim() != pad_width.len()`.
fn pad_with_zeros<A, S, D>(arr: &ArrayBase<S, D>, pad_width: Vec<[usize; 2]>) -> Array<A, D>
where
    A: Clone + Zero,
    S: Data<Elem = A>,
    D: Dimension,
{
    assert_eq!(
        arr.ndim(),
        pad_width.len(),
        "Array ndim must match length of `pad_width`."
    );

    // Compute shape of final padded array.
    let mut padded_shape = arr.raw_dim();
    for (ax, (&ax_len, &[pad_lo, pad_hi])) in arr.shape().iter().zip(&pad_width).enumerate() {
        println!("ax - {ax}, ax_len - {ax_len}, pad_lo - {pad_lo}, pad-hi - {pad_hi}");
        println!("ax_up - {}", ax_len + pad_lo + pad_hi);
        padded_shape[ax] = ax_len + pad_lo + pad_hi;
    }

    let mut padded = Array::zeros(padded_shape);
    {
        // Select portion of padded array that needs to be copied from the
        // original array.
        let mut orig_portion = padded.view_mut();
        for (ax, &[pad_lo, pad_hi]) in pad_width.iter().enumerate() {
            // FIXME: This has a bug when `pad_hi` is 0. See @fzyzcjy's comment below.
            orig_portion
                .slice_axis_inplace(Axis(ax), Slice::from(pad_lo as isize..-(pad_hi as isize)));
        }
        // Copy the data from the original array.
        orig_portion.assign(arr);
    }
    padded
}

fn main() {
    use ndarray::Array2;
    let to_pad = Array2::<f32>::ones((3, 4));
    println!("{}", to_pad);
    //let padded = pad_with_zeros(&to_pad, vec![[2, 1], [1, 1]]);
    //println!("{}", padded);
}
