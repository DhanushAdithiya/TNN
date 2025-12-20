use ndarray::ShapeBuilder;
use ndarray::{Array, ArrayD};
use std::f32::consts::E;
use std::fmt;
use std::usize;

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

    pub fn matmul(&self, other: Tensor) -> Result<Tensor, ShapeError> {
        Tensor::check_compatible(self, &other)?;

        let a = Tensor::to_contiguous(&self.data);
        let b = Tensor::to_contiguous(&other.data);

        let m = self.shape()[0]; // rows of A
        let k = self.shape()[1]; // cols of A = rows of B
        let n = other.shape()[1]; // cols of B

        let a_slice = a.as_slice().unwrap();
        let b_slice = b.as_slice().unwrap();

        let mut out = vec![0.0f32; m * n];

        // i-k-j loop (best cache behavior)
        for i in 0..m {
            let out_row = &mut out[i * n..(i + 1) * n];
            let a_row = &a_slice[i * k..(i + 1) * k];

            for kk in 0..k {
                let a_val = a_row[kk];
                let b_row = &b_slice[kk * n..(kk + 1) * n];

                for j in 0..n {
                    out_row[j] += a_val * b_row[j];
                }
            }
        }

        Ok(Tensor::from(&[m, n], out, false))
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
