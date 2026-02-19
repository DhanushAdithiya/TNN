#[cfg(test)]
use crate::tensor::{Tensor};
use crate::raw_tensor::RawTensor;
use ndarray::array;

#[test]
fn test_tensor_add() {
    let x = Tensor::from(RawTensor::from(
        &[2, 2],
        vec![1., 2., 3., 4.],
        false, // col_major: false
    ));

    let y = &x + &x;

    // Forward: [[1+1, 2+2], [3+3, 4+4]]
    let expected = array![[2., 4.], [6., 8.]].into_dyn();
    assert_eq!(y.inner.borrow().tensor.data.view(), expected);

    y.backward();

    // Gradient: d/dx(x+x) = 2
    let grad_expected = array![[2., 2.], [2., 2.]].into_dyn();
    assert_eq!(
        x.inner.borrow().gradient.as_ref().unwrap().data.view(),
        grad_expected
    );
}

#[test]
fn test_tensor_sub() {
    let x = Tensor::from(RawTensor::from(
        &[2, 2],
        vec![1., 2., 3., 4.],
        false, // col_major: false
    ));

    let y = &x - &x;

    let expected = array![[0., 0.], [0., 0.]].into_dyn();
    assert_eq!(y.inner.borrow().tensor.data.view(), expected);

    y.backward();

    let grad_expected = array![[0., 0.], [0., 0.]].into_dyn();
    assert_eq!(
        x.inner.borrow().gradient.as_ref().unwrap().data.view(),
        grad_expected
    );
}

#[test]
fn test_tensor_mul() {
    let x = Tensor::from(RawTensor::from(
        &[2, 2],
        vec![1., 2., 3., 4.],
        false, // col_major: false
    ));

    let y = &x * &x;

    // Forward: [[1*1, 2*2], [3*3, 4*4]]
    let expected = array![[1., 4.], [9., 16.]].into_dyn();
    assert_eq!(y.inner.borrow().tensor.data.view(), expected);

    y.backward();

    // Gradient: d/dx(x^2) = 2x -> [[2*1, 2*2], [2*3, 2*4]]
    let grad_expected = array![[2., 4.], [6., 8.]].into_dyn();
    assert_eq!(
        x.inner.borrow().gradient.as_ref().unwrap().data.view(),
        grad_expected
    );
}

#[test]
fn test_tensor_div_self() {
    let x = Tensor::from(RawTensor::from(
        &[2, 2],
        vec![1., 2., 3., 4.],
        false, // col_major: false
    ));

    let y = &x / &x;

    let expected = array![[1., 1.], [1., 1.]].into_dyn();
    assert_eq!(y.inner.borrow().tensor.data.view(), expected);

    y.backward();

    let grad_expected = array![[0., 0.], [0., 0.]].into_dyn();
    assert_eq!(
        x.inner.borrow().gradient.as_ref().unwrap().data.view(),
        grad_expected
    );
}

#[test]
fn test_tensor_div_reciprocal() {
    let x = Tensor::from(RawTensor::from(
        &[2, 2],
        vec![1., 2., 3., 4.],
        false, // col_major: false
    ));

    let one = Tensor::from(RawTensor::from(
        &[2, 2],
        vec![1., 1., 1., 1.],
        false, // col_major: false
    ));

    let y = &one / &x;

    y.backward();

    let grad = x.inner.borrow().gradient.as_ref().unwrap().data.clone();

    // d/dx(1/x) = -1 / x^2
    // Row 0: -1/1, -1/4
    // Row 1: -1/9, -1/16
    let expected = array![
        [-1.0, -0.25],
        [-1.0/9.0, -0.0625]
    ].into_dyn();

    for (a, b) in grad.iter().zip(expected.iter()) {
        assert!((a - b).abs() < 1e-6);
    }
}

// #[test]
// fn test_tensor_neg() {
//     let x = Tensor::from(RawTensor::from(
//         &[2, 2],
//         vec![1., 2., 3., 4.],
//         true,
//     ));

//     let y = -&x;

//     let expected = array![[-1., -2.], [-3., -4.]].into_dyn();
//     assert_eq!(y.inner.borrow().tensor.data.view(), expected);

//     y.backward();

//     let grad_expected = array![[-1., -1.], [-1., -1.]].into_dyn();
//     assert_eq!(
//         x.inner.borrow().gradient.as_ref().unwrap().data.view(),
//         grad_expected
//     );
// }
