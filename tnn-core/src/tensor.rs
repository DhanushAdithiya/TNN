use ndarray::Array2;
use ndarray::IxDyn;
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

    pub fn from(shape: &[usize], data: Vec<f32>) -> Tensor {
        let data = Array::from_shape_vec(shape, data).expect("Shape doesnt match data");
        return Tensor { data };
    }

    pub fn shape(&self) -> &[usize] {
        return self.data.shape();
    }

    pub fn zeros(shape: &[usize]) -> Tensor {
        let len = shape[0] * shape[1];
        return Tensor::from(shape, vec![0.0; len]);
    }

    pub fn reshape(&mut self, shape: &[usize]) {
        let data: Vec<f32> = self.data.clone().into_iter().collect();
        self.data = Array::from_shape_vec(shape, data.to_owned())
            .expect("Could not recast due to shape error");
    }

    fn to_column_major(a: &ArrayD<f32>) -> ArrayD<f32> {
        let dims = a.dim();
        let mut col_major = ArrayD::<f32>::zeros(dims.f()); // Fortran / column-major layout
        col_major.assign(a); // copies data, changing layout
        col_major
    }

    pub fn matmul(&self, other: Tensor) -> Result<Tensor, ShapeError> {
        match Tensor::check_compatible(&self, &other) {
            Ok(()) => {
                let mut data = Vec::new();
                let other = Tensor::to_column_major(&other.data);
                for rows in self.data.rows() {
                    for cols in other.axis_iter(ndarray::Axis(1)).to_owned() {
                        //println!("transposed - {:?}", cols);
                        //for val in &cols {
                        //    let addr: *const f32 = val as *const f32;
                        //    println!("Address : {:p}", addr);
                        //}
                        let val: f32 = std::iter::zip(rows, &cols).map(|(a, b)| *a * *b).sum();
                        data.push(val);
                    }
                }

                return Ok(Tensor::from(&[self.shape()[0], other.shape()[1]], data));
            }

            Err(e) => {
                eprintln!("Could not multiply matrices");
                return Err(e);
            }
        }
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
        return Tensor { data: t_clone };
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
