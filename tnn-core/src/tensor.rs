use ndarray::{s, ShapeBuilder};
use ndarray::{Array, ArrayD};
use std::f32::consts::E;
use std::fmt;
use std::thread;
use std::usize;

// TODO
// [] - Remove Column Order
// [] - Change ReLu , Add to use array slices

#[derive(Clone, Debug)]
pub struct ShapeError {
    left: Vec<usize>,
    right: Vec<usize>,
}

impl fmt::Display for ShapeError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Matrices of invalid size {:?}-{:?}",
            self.left, self.right
        )
    }
}

impl std::error::Error for ShapeError {}

#[derive(Clone)]
pub struct Tensor {
    pub data: ArrayD<f32>,
    pub column_major: bool,
}

impl Tensor {
    fn check_compatible(first: &Tensor, second: &Tensor) -> Result<(), ShapeError> {
        // FIX : Dont need to send the entire tensor here, only the shapes would do

        if first.shape()[0] != second.shape()[1] {
            return Err(ShapeError {
                left: first.shape().to_vec(),
                right: second.shape().to_vec(),
            });
        }

        return Ok(());
    }

    pub fn from(shape: &[usize], data: Vec<f32>, col_major: bool) -> Tensor {
        let mut data = Array::from_shape_vec(shape, data).expect("Shape doesnt match data");

        if col_major {
            data = Tensor::to_column_major(&data);
            return Tensor {
                data,
                column_major: true,
            };
        }

        return Tensor {
            data,
            column_major: false,
        };
    }

    pub fn shape(&self) -> &[usize] {
        return self.data.shape();
    }

    pub fn zeros(shape: &[usize]) -> Tensor {
        let len = shape[0] * shape[1];
        return Tensor::from(shape, vec![0.0; len], true);
    }

    pub fn reshape(&mut self, shape: &[usize]) {
        let data: Vec<f32> = self.data.clone().into_iter().collect();
        self.data = Array::from_shape_vec(shape, data.to_owned())
            .expect("Could not recast due to shape error");
    }

    fn to_contiguous(a: &ArrayD<f32>) -> ArrayD<f32> {
        if let Some(_) = a.as_slice() {
            return a.clone();
        };
        return a.to_owned();
    }

    fn to_column_major(a: &ArrayD<f32>) -> ArrayD<f32> {
        let dims = a.dim();
        let mut col_major = ArrayD::<f32>::zeros(dims.f()); // Fortran / column-major layout
        col_major.assign(a); // copies data, changing layout
        col_major
    }

    pub fn block_mul_at(
        a: &[f32],
        b: &[f32],
        out_ptr: &mut [f32],
        m: usize,
        k: usize,
        n: usize,
        total_stride: usize,
    ) {
        for i in 0..m {
            let a_row = &a[i * k..(i + 1) * k];
            let row_start = i * total_stride;
            // Get a mutable slice for the specific row segment we are working on
            let out_row = &mut out_ptr[row_start..row_start + n];

            for j in 0..k {
                let a_val = a_row[j];
                let b_row = &b[j * n..(j + 1) * n];

                for l in 0..n {
                    out_row[l] += a_val * b_row[l];
                }
            }
        }
    }

    pub fn matmul(&self, other: Tensor) -> Result<Tensor, ShapeError> {
        Tensor::check_compatible(self, &other)?;

        let m = self.shape()[0];
        let k = self.shape()[1];
        let n = other.shape()[1];

        let mut out_data = vec![0.0f32; m * n];

        let m_mid = m / 2;
        let k_mid = k / 2;
        let n_mid = n / 2;

        // --- Prepare Contiguous Slices for A and B ---
        // We to_owned() these to ensure they are contiguous slices for block_mul_at
        let a11 = self.data.slice(s![0..m_mid, 0..k_mid]).to_owned();
        let a12 = self.data.slice(s![0..m_mid, k_mid..]).to_owned();
        let a21 = self.data.slice(s![m_mid.., 0..k_mid]).to_owned();
        let a22 = self.data.slice(s![m_mid.., k_mid..]).to_owned();

        let b11 = other.data.slice(s![0..k_mid, 0..n_mid]).to_owned();
        let b12 = other.data.slice(s![0..k_mid, n_mid..]).to_owned();
        let b21 = other.data.slice(s![k_mid.., 0..n_mid]).to_owned();
        let b22 = other.data.slice(s![k_mid.., n_mid..]).to_owned();

        // Convert to slices once to avoid calling as_slice() repeatedly inside threads
        let a11_s = a11.as_slice().unwrap();
        let a12_s = a12.as_slice().unwrap();
        let a21_s = a21.as_slice().unwrap();
        let a22_s = a22.as_slice().unwrap();

        let b11_s = b11.as_slice().unwrap();
        let b12_s = b12.as_slice().unwrap();
        let b21_s = b21.as_slice().unwrap();
        let b22_s = b22.as_slice().unwrap();

        // 1. Split into Top rows and Bottom rows
        // This is the key to satisfying the borrow checker for 2 threads
        let (top_rows, bottom_rows) = out_data.split_at_mut(m_mid * n);

        thread::scope(|s| {
            // --- Thread 1: Handles Top Half (C11 and C12) ---
            s.spawn(|| {
                // C11 = A11*B11 + A12*B21
                Self::block_mul_at(a11_s, b11_s, top_rows, m_mid, k_mid, n_mid, n);
                Self::block_mul_at(a12_s, b21_s, top_rows, m_mid, k - k_mid, n_mid, n);

                // C12 = A11*B12 + A12*B22
                let c12_view = &mut top_rows[n_mid..];
                Self::block_mul_at(a11_s, b12_s, c12_view, m_mid, k_mid, n - n_mid, n);
                Self::block_mul_at(a12_s, b22_s, c12_view, m_mid, k - k_mid, n - n_mid, n);
            });

            // --- Thread 2: Handles Bottom Half (C21 and C22) ---
            s.spawn(|| {
                // C21 = A21*B11 + A22*B21
                Self::block_mul_at(a21_s, b11_s, bottom_rows, m - m_mid, k_mid, n_mid, n);
                Self::block_mul_at(a22_s, b21_s, bottom_rows, m - m_mid, k - k_mid, n_mid, n);

                // C22 = A21*B12 + A22*B22
                let c22_view = &mut bottom_rows[n_mid..];
                Self::block_mul_at(a21_s, b12_s, c22_view, m - m_mid, k_mid, n - n_mid, n);
                Self::block_mul_at(a22_s, b22_s, c22_view, m - m_mid, k - k_mid, n - n_mid, n);
            });
        });

        Ok(Tensor::from(&[m, n], out_data, false))
    }

    pub fn add(&mut self, other: Tensor) -> Tensor {
        assert_eq!(
            self.shape(),
            other.shape(),
            "Tensors with different shapes cannot be added {:?} + {:?}",
            self.shape(),
            other.shape()
        );

        let mut t_clone = self.data.clone();
        t_clone.scaled_add(1., &other.data);
        return Tensor {
            data: t_clone,
            column_major: true,
        };
    }

    pub fn relu(&mut self) {
        self.data.iter_mut().for_each(|x| {
            if *x < 0. {
                *x = 0.;
            }
        });
    }

    pub fn sigmoid(&mut self) {
        self.data.iter_mut().for_each(|x| {
            *x = Tensor::sigx(*x);
        });
    }

    fn sigx(x: f32) -> f32 {
        return 1. / (1. + E.powf(-x));
    }
}
