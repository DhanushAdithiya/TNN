use tnn_core::tensor::Tensor;
use ndarray::Axis;


fn main() {
    let tensor = Tensor::zeros(&[2,3]);
    println!("{:?}",tensor.data);
    let data = tensor.data.select(Axis(0), &[0]);
    println!("{:?}",tensor.data.view());
}
