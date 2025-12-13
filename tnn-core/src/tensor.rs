use std::usize;

use ndarray::{Array, Array2, ArrayD};

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
}

#[cfg(test)]
mod tests {
    use crate::tensor::Tensor;

    #[test]
    fn test_create_tensor() {
        let data = vec![1.0, 2.0, 3.0, 3.0, 5.4444, 23.444];
        let tensor = Tensor::from(&[2, 3], data);

        assert_eq!(tensor.shape(), [2, 3]);
    }

    #[test]
    fn test_tensor_shape() {
        let data = vec![1.0, 2.0, 3.0, 3.0, 5.4444, 23.444];
        let tensor = Tensor::from(&[2, 3], data);
        assert_eq!(tensor.shape(), [2, 3]);
    }

    #[test]
    #[should_panic]
    fn test_shape_mismatch() {
        let data = vec![1.0, 2.0, 3.0, 3.0, 5.4444, 23.444];
        let _tensor = Tensor::from(&[2, 4], data);
    }

    #[test]
    fn test_zeros() {
        let tensor = Tensor::zeros(&[2, 3]);
        tensor.data.iter().for_each(|x| assert_eq!(*x, 0.0));
    }

    #[test]
    fn test_reshape() {
        let data = vec![1.0, 2.0, 3.0, 3.0, 5.4444, 23.444];
        let mut tensor = Tensor::from(&[2, 3], data);

        tensor.reshape(&[3, 2]);
        assert_eq!(tensor.shape(), [3, 2]);
    }

    #[test]
    #[should_panic]
    fn test_reshape_panic() {
        let data = vec![1.0, 2.0, 3.0, 3.0, 5.4444, 23.444];
        let mut tensor = Tensor::from(&[2, 3], data);

        tensor.reshape(&[3, 3]);
    }

    #[test]
    fn test_matmul() {
        let data1 = vec![1., 2., 3., 4.];
        let data2 = vec![1., 2., 3., 4.];

        let t1 = Tensor::from(&[2, 2], data1);
        let t2 = Tensor::from(&[2, 2], data2);

        let t3 = t1.matmul(t2);
        use ndarray::array;

        assert_eq!(
            t3.data.view(),
            array![[7., 10.], [15., 22.]].view().into_dyn()
        );
    }

    #[test]
    fn test_matmul2() {
        // A 6×6 matrix
        let data1 = vec![
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19.,
            20., 21., 22., 23., 24., 25., 26., 27., 28., 29., 30., 31., 32., 33., 34., 35., 36.,
        ];

        // Another 6×6 matrix
        let data2 = vec![
            6., 5., 4., 3., 2., 1., 12., 11., 10., 9., 8., 7., 18., 17., 16., 15., 14., 13., 24.,
            23., 22., 21., 20., 19., 30., 29., 28., 27., 26., 25., 36., 35., 34., 33., 32., 31.,
        ];

        let t1 = Tensor::from(&[6, 6], data1);
        let t2 = Tensor::from(&[6, 6], data2);

        let t3 = t1.matmul(t2);

        use ndarray::array;

        let expected = array![
            [546., 525., 504., 483., 462., 441.],
            [1302., 1245., 1188., 1131., 1074., 1017.],
            [2058., 1965., 1872., 1779., 1686., 1593.],
            [2814., 2685., 2556., 2427., 2298., 2169.],
            [3570., 3405., 3240., 3075., 2910., 2745.],
            [4326., 4125., 3924., 3723., 3522., 3321.],
        ]
        .into_dyn();

        assert_eq!(t3.data.view(), expected.view());
    }

    #[test]
    #[should_panic]
    fn test_matmul_fail() {
        // A 6×6 matrix
        let data1 = vec![
            7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24.,
            25., 26., 27., 28., 29., 30., 31., 32., 33., 34., 35., 36.,
        ];

        // Another 6×6 matrix
        let data2 = vec![
            6., 5., 4., 3., 2., 1., 12., 11., 10., 9., 8., 7., 18., 17., 16., 15., 14., 13., 24.,
            23., 22., 21., 20., 19., 30., 29., 28., 27., 26., 25., 36., 35., 34., 33., 32., 31.,
        ];

        let t1 = Tensor::from(&[5, 6], data1);
        let t2 = Tensor::from(&[6, 6], data2);

        let t3 = t1.matmul(t2);
    }
}
