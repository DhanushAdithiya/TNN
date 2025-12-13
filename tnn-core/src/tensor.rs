use ndarray::{Array, ArrayD};
use std::f32::consts::E;
use std::usize;

pub struct Tensor {
    pub data: ArrayD<f32>,
}

impl Tensor {
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

    pub fn matmul(&self, other: Tensor) -> Tensor {
        assert_eq!(
            self.shape()[0],
            other.shape()[1],
            "Cannot matrix multiply matrices of shapes {}, {}",
            self.shape()[0],
            other.shape()[1]
        );

        let mut data = Vec::new();

        for rows in self.data.rows() {
            for cols in other.data.columns() {
                let val: f32 = std::iter::zip(rows, cols).map(|(a, b)| *a * *b).sum();
                data.push(val);
            }
        }

        return Tensor::from(&[self.shape()[0], other.shape()[1]], data);
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
