use crate::raw_tensor::RawTensor;
use std::cell::RefCell;
use std::ops::{Add, Div, Mul, Neg, Sub};

type BackwardFn = fn(AutogradNode, AutogradNode, AutogradNode);

#[derive(Debug, Clone)]
pub struct Tensor<'a> {
    pub tensor: RawTensor,
    gradient: Option<RawTensor>,
    node: Option<AutogradNode<'a>>,
}

#[derive(Debug, Clone)]
pub struct AutogradNode<'a> {
    parents: RefCell<Vec<Tensor<'a>>>,
    sign: String,
    // backwards: BackwardFn
}

impl<'a, 'b> Add<&'b Tensor> for &'a Tensor {
    type Output = Tensor;

    fn add(self, rhs: &'b Tensor) -> Tensor {
        let parents = RefCell::new(vec![self, rhs]);

        let node = AutogradNode {
            parents,
            sign: "+".to_string(),
        };

        let o = self.tensor.add(&rhs.tensor);

        Tensor {
            tensor: o,
            gradient: None,
            node: Some(node),
        }
    }
}

impl<'a, 'b> Sub<&'b Tensor> for &'a Tensor {
    type Output = Tensor;

    fn sub(self, rhs: &'b Tensor) -> Tensor {
        let parents = RefCell::from(vec![self.clone(), rhs.clone()]);
        let node = AutogradNode {
            parents,
            sign: "-".to_string(),
        };

        let o = self.tensor.sub(&rhs.tensor);
        return Tensor {
            tensor: o,
            gradient: None,
            node: Some(node),
        };
    }
}

impl<'a, 'b> Mul<&'b Tensor> for &'a Tensor {
    type Output = Tensor;

    fn mul(self, rhs: &'b Tensor) -> Tensor {
        let o = self.tensor.mul(&rhs.tensor);
        let parents = RefCell::from(vec![self.clone(), rhs.clone()]);
        let node = AutogradNode {
            parents,
            sign: "*".to_string(),
        };
        return Tensor {
            tensor: o,
            gradient: None,
            node: Some(node),
        };
    }
}

impl<'a, 'b> Div<&'b Tensor> for &'a Tensor {
    type Output = Tensor;

    fn div(self, rhs: &'b Tensor) -> Tensor {
        let o = self.tensor.div(&rhs.tensor);
        let parents = RefCell::from(vec![self.clone(), rhs.clone()]);
        let node = AutogradNode {
            parents,
            sign: "-".to_string(),
        };
        return Tensor {
            tensor: o,
            gradient: None,
            node: Some(node),
        };
    }
}

impl Neg for Tensor {
    type Output = Self;

    fn neg(mut self) -> Self::Output {
        return Tensor {
            tensor: self.tensor.scale(-1.0),
            gradient: self.gradient,
            node: self.node,
        };
    }
}

impl Tensor {
    pub fn from(raw_tensor: RawTensor) -> Self {
        let grad = RawTensor::zeros(raw_tensor.shape());
        return {
            Tensor {
                tensor: raw_tensor,
                gradient: Some(grad),
                node: None,
            }
        };
    }

    pub fn relu(mut self) -> Self {
        self.tensor.relu();
        return Tensor {
            tensor: self.tensor,
            gradient: self.gradient,
            node: self.node,
        };
    }

    pub fn sigmoid(mut self) -> Self {
        self.tensor.sigmoid();
        return Tensor {
            tensor: self.tensor,
            gradient: self.gradient,
            node: self.node,
        };
    }

    pub fn pow(&self, exp: f32) -> Self {
        let o = self.tensor.data.powf(exp);
        return Tensor {
            tensor: RawTensor {
                data: o,
                column_major: false,
            },
            gradient: None,
            node: None,
        };
    }

    pub fn backwards(&mut self) {
        todo!()
    }
}
