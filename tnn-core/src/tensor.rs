use crate::raw_tensor::RawTensor;
use std::ops::{Add, Mul, Sub, Div, Neg};

type BackwardFn = fn(AutogradNode, AutogradNode, AutogradNode);

#[derive(Debug, Clone)]
pub struct Tensor {
    tensor: RawTensor,
    gradient: Option<RawTensor>,
    node: Option<AutogradNode>
}

#[derive(Debug, Clone)]
pub struct AutogradNode {
    parents: Vec<Tensor>,
    backwards: BackwardFn
}

impl Add for Tensor {
    type Output = Self;
    
    fn add(mut self, rhs: Self) -> Self::Output {
        let o = self.tensor.add(rhs.tensor);
        return Tensor {
            tensor: o,
            gradient: self.gradient,
            node: self.node
        };
    }
    
}

impl Sub for Tensor {
    type Output = Self;
    fn sub(mut self, rhs: Self) -> Self::Output {
        let o = self.tensor.sub(rhs.tensor);
        return Tensor {
            tensor: o,
            gradient: self.gradient,
            node: self.node
        };
    }
}

impl Mul for Tensor {
    type Output = Self;

    fn mul(mut self, rhs: Self) -> Self::Output {
        let o = self.tensor.mul(rhs.tensor);
        return Tensor {
            tensor: o,
            gradient: self.gradient,
            node: self.node
        };
    }
}

impl Div for Tensor {
    type Output = Self;

    fn div(mut self, rhs: Self) -> Self::Output {
        let o = self.tensor.div(rhs.tensor);
        return Tensor {
            tensor: o,
            gradient: self.gradient,
            node: self.node
        }
    }
}

impl Neg for Tensor {
    type Output = Self;

    fn neg(mut self) -> Self::Output {
        return Tensor {
            tensor: self.tensor.scale(-1.0),
            gradient: self.gradient,
            node: self.node
        }
    }
}

impl Tensor {
    pub fn from(raw_tensor: RawTensor) -> Self {
        let grad = RawTensor::zeros(raw_tensor.shape());
        return {
            Tensor { tensor: raw_tensor, gradient: Some(grad), node: None }
        }
    }

    pub fn relu(mut self) -> Self {
        self.tensor.relu();
        return Tensor {
            tensor: self.tensor,
            gradient: self.gradient,
            node: self.node
        }
    }

    pub fn sigmoid(mut self) -> Self {
        self.tensor.sigmoid();
        return  Tensor { tensor: self.tensor, gradient: self.gradient, node: self.node };
    }

    pub fn pow(self, exp: f32) -> Self {
        let o = self.tensor.data.powf(exp);
        return Tensor {
            tensor: RawTensor { data: o, column_major: false },
            gradient: self.gradient,
            node: self.node
        }
    }

    pub fn backwards(&mut self) {
        todo!()
    }
}