use tnn_core::tensor::Tensor;

fn main() {
    let mut t1 = Tensor::from(&[2, 2], [1., 2., 3., 4.].to_vec());
    let t2 = Tensor::from(&[3, 2], [1., 2., 3., 4., 3., 4.].to_vec());

    t1.add(t2);
}
