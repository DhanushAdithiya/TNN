#![allow(unused)]
use crate::raw_tensor::RawTensor;
use std::cell::RefCell;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::rc::Rc;

type BackwardFn = fn(AutogradNode, AutogradNode, AutogradNode);
type ParentsRef = Vec<Rc<RefCell<Tensor>>>;

#[derive(Clone, Debug)]
enum Op {
    Add,
    Sub,
    Mul,
    Div,
    Neg,
    Relu,
}

#[derive(Debug, Clone)]
pub struct Tensor {
    pub tensor: RawTensor,
    pub gradient: Option<RawTensor>,
    node: Option<AutogradNode>,
}

#[derive(Debug, Clone)]
pub struct AutogradNode {
    parents: ParentsRef,
    sign: Op,
    backwards: BackwardFn,
}


let backwards_add = |v1: Tensor, v2: Tensor, o: Tensor| {
    v1.gradient.unwrap().add(v2.gradient.unwrap().mul(&o.gradint.clone().unwrap()))
    v2.gradient.unwrap().add(v1.gradient.unwrap().mul(&o.gradint.clone().unwrap()))
};

impl<'a, 'b> Add<&'b Tensor> for &'a Tensor {
    type Output = Tensor;

    fn add(self, rhs: &'b Tensor) -> Tensor {
        let parents = vec![
            Rc::new(RefCell::new(self.clone())),
            Rc::new(RefCell::new(rhs.clone())),
        ];

        let o = self.tensor.add(&rhs.tensor);
        let mut op = Tensor {
            tensor: o,
            gradient: Some(RawTensor::zeros(self.tensor.shape())),
            node: None,
        };

        let node = AutogradNode {
            parents,
            sign: Op::Add,
            backwards: backwards_add(self, rhs, op),
        };

        op.node = Some(node);
        return op;
    }
}

impl<'a, 'b> Sub<&'b Tensor> for &'a Tensor {
    type Output = Tensor;

    fn sub(self, rhs: &'b Tensor) -> Tensor {
        let parents = vec![
            Rc::new(RefCell::new(self.clone())),
            Rc::new(RefCell::new(rhs.clone())),
        ];

        let node = AutogradNode {
            parents,
            sign: Op::Sub,
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
        let parents = vec![
            Rc::new(RefCell::new(self.clone())),
            Rc::new(RefCell::new(rhs.clone())),
        ];
        let node = AutogradNode {
            parents,
            sign: Op::Mul,
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
        let parents = vec![
            Rc::new(RefCell::new(self.clone())),
            Rc::new(RefCell::new(rhs.clone())),
        ];

        let node = AutogradNode {
            parents,
            sign: Op::Div,
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
        self.gradient = Some(RawTensor::ones(self.tensor.shape()))
    }
}
