#![allow(unused)]
use crate::raw_tensor::RawTensor;
use std::cell::RefCell;
use std::collections::BTreeSet;
use std::collections::HashSet;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::rc::Rc;

type BackwardFn = fn(Tensor, Tensor, Tensor);
type ParentsRef = Vec<Rc<RefCell<TensorData>>>;

#[derive(Clone, Debug, Hash)]
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
    pub inner: Rc<RefCell<TensorData>>,
}

#[derive(Debug, Clone)]
pub struct TensorData {
    pub tensor: RawTensor,
    pub gradient: Option<RawTensor>,
    node: Option<AutogradNode>,
}

#[derive(Debug, Clone)]
pub struct AutogradNode {
    parents: Vec<Tensor>,
    sign: Op,
    backwards: BackwardFn,
}

fn backwards_add(lhs: Tensor, rhs: Tensor, o: Tensor) {
    let o_grad = o.inner.borrow().gradient.as_ref().unwrap().clone();

    {
        let mut lhs_data = lhs.inner.borrow_mut();
        if let Some(ref mut lhs_grad) = lhs_data.gradient {
            *lhs_grad = lhs_grad.add(&o_grad);
        }
    }

    {
        let mut rhs_data = rhs.inner.borrow_mut();
        if let Some(ref mut rhs_grad) = rhs_data.gradient {
            *rhs_grad = rhs_grad.add(&o_grad);
        }
    }
}

fn backwards_sub(lhs: Tensor, rhs: Tensor, o: Tensor) {
    let o_grad = o.inner.borrow().gradient.as_ref().unwrap().clone();
    
    if Rc::ptr_eq(&lhs.inner, &rhs.inner) {
        // Same tensor - combine gradients
        let mut data = lhs.inner.borrow_mut();
        if let Some(ref mut grad) = data.gradient {
            *grad = grad.add(&o_grad).add(&o_grad.neg());
            // or: *grad = grad.add(&o_grad).sub(&o_grad) = no change
        }
    } else {
        // Different tensors
        let mut lhs_data = lhs.inner.borrow_mut();
        if let Some(ref mut lhs_grad) = lhs_data.gradient {
            *lhs_grad = lhs_grad.add(&o_grad);
        }
        drop(lhs_data);
        
        let mut rhs_data = rhs.inner.borrow_mut();
        if let Some(ref mut rhs_grad) = rhs_data.gradient {
            *rhs_grad = rhs_grad.add(&o_grad.neg());
        }
    }
}

fn backwards_mul(lhs: Tensor, rhs: Tensor, o: Tensor) {
    let o_grad = o.inner.borrow().gradient.as_ref().unwrap().clone();

    let mut lhs_data = lhs.inner.borrow_mut();
    let mut rhs_data = rhs.inner.borrow_mut();

    if let Some(ref mut lhs_grad) = lhs_data.gradient {
        let grad = rhs_data.tensor.mul(&o_grad);
        *lhs_grad = lhs_grad.add(&grad);
    }

    if let Some(ref mut rhs_grad) = rhs_data.gradient {
        let grad = lhs_data.tensor.mul(&o_grad);
        *rhs_grad = rhs_grad.add(&grad);
    }
}

fn backwards_div(lhs: Tensor, rhs: Tensor, o: Tensor) {
    let o_grad = o.inner.borrow().gradient.as_ref().unwrap().clone();

    let mut lhs_data = lhs.inner.borrow_mut();
    let mut rhs_data = rhs.inner.borrow_mut();
    let lhs_shape = lhs_data.tensor.shape().to_owned();
    let rhs_shape = rhs_data.tensor.shape().to_owned();

    if let Some(ref mut lhs_grad) = lhs_data.gradient {
        let ones = RawTensor::ones(&lhs_shape);
        let grad = ones.div(&rhs_data.tensor).mul(&o_grad);
        *lhs_grad = lhs_grad.add(&grad);
    }

    if let Some(ref mut rhs_grad) = rhs_data.gradient {
        let ones = RawTensor::ones(&rhs_shape);
        let grad = ones.div(&lhs_data.tensor).mul(&o_grad);
        *rhs_grad = rhs_grad.add(&grad);
        println!("rHS - DIV - {:?}", rhs_grad);
    }
}

impl<'a> Add<&'a Tensor> for &'a Tensor {
    type Output = Tensor;

    fn add(self, rhs: &'a Tensor) -> Self::Output {
        let lhs_data = self.inner.borrow();
        let rhs_data = rhs.inner.borrow();

        let o = lhs_data.tensor.add(&rhs_data.tensor);
        let shape = lhs_data.tensor.shape();

        drop(rhs_data);

        let node = AutogradNode {
            parents: vec![self.clone(), rhs.clone()],
            sign: Op::Add,
            backwards: backwards_add,
        };

        Tensor {
            inner: Rc::new(RefCell::new(TensorData {
                tensor: o,
                gradient: Some(RawTensor::zeros(shape)),
                node: Some(node),
            })),
        }
    }
}

impl<'a> Sub<&'a Tensor> for &'a Tensor {
    type Output = Tensor;
    fn sub(self, rhs: &'a Tensor) -> Self::Output {
        let lhs_data = self.inner.borrow();
        let rhs_data = rhs.inner.borrow();

        let o = lhs_data.tensor.sub(&rhs_data.tensor);
        let shape = lhs_data.tensor.shape();

        drop(rhs_data);

        let node = AutogradNode {
            parents: vec![self.clone(), rhs.clone()],
            backwards: backwards_sub,
            sign: Op::Sub,
        };

        Tensor {
            inner: Rc::new(RefCell::new(TensorData {
                gradient: Some(RawTensor::zeros(shape)),
                node: Some(node),
                tensor: o,
            })),
        }
    }
}

impl<'a> Div<&'a Tensor> for &'a Tensor {
    type Output = Tensor;
    fn div(self, rhs: &'a Tensor) -> Self::Output {
        let lhs_data = self.inner.borrow();
        let rhs_data = rhs.inner.borrow();

        let o = lhs_data.tensor.div(&rhs_data.tensor);
        let shape = lhs_data.tensor.shape();

        drop(rhs_data);
        let node = AutogradNode {
            parents: vec![self.clone(), rhs.clone()],
            backwards: backwards_div,
            sign: Op::Div,
        };

        Tensor {
            inner: Rc::new(RefCell::new(TensorData {
                tensor: o,
                gradient: Some(RawTensor::zeros(shape)),
                node: Some(node),
            })),
        }
    }
}

impl<'a> Mul<&'a Tensor> for &'a Tensor {
    type Output = Tensor;
    fn mul(self, rhs: &'a Tensor) -> Self::Output {
        let lhs_data = self.inner.borrow();
        let rhs_data = rhs.inner.borrow();

        let o = lhs_data.tensor.mul(&rhs_data.tensor);
        let shape = lhs_data.tensor.shape();

        let node = AutogradNode {
            backwards: backwards_mul,
            parents: vec![self.clone(), rhs.clone()],
            sign: Op::Mul,
        };

        Tensor {
            inner: Rc::new(RefCell::new(TensorData {
                gradient: Some(RawTensor::zeros(shape)),
                node: Some(node),
                tensor: o,
            })),
        }
    }
}

impl Neg for Tensor {
    type Output = Self;

    fn neg(self) -> Self::Output {
        let mut lhs_data = self.inner.borrow_mut();
        Tensor {
            inner: Rc::new(RefCell::new(TensorData {
                tensor: lhs_data.tensor.scale(-1.0),
                gradient: lhs_data.gradient.clone(),
                node: lhs_data.node.clone(),
            })),
        }
    }
}

impl TensorData {
    pub fn from(raw_tensor: RawTensor) -> Self {
        let grad = RawTensor::zeros(raw_tensor.shape());
        return {
            TensorData {
                tensor: raw_tensor,
                gradient: Some(grad),
                node: None,
            }
        };
    }

    pub fn relu(mut self) -> Self {
        self.tensor.relu();
        return TensorData {
            tensor: self.tensor,
            gradient: self.gradient,
            node: self.node,
        };
    }

    pub fn sigmoid(mut self) -> Self {
        self.tensor.sigmoid();
        return TensorData {
            tensor: self.tensor,
            gradient: self.gradient,
            node: self.node,
        };
    }

    pub fn pow(&self, exp: f32) -> Self {
        let o = self.tensor.data.powf(exp);
        return TensorData {
            tensor: RawTensor {
                data: o,
                column_major: false,
            },
            gradient: None,
            node: None,
        };
    }
}

impl Tensor {
    pub fn from(raw_tensor: RawTensor) -> Self {
        let grad = RawTensor::zeros(raw_tensor.shape());
        Tensor {
            inner: Rc::new(RefCell::new(TensorData {
                tensor: raw_tensor,
                gradient: Some(grad),
                node: None,
            })),
        }
    }

    pub fn backward(&self) {
        // Set this tensor's gradient to 1.0
        {
            let mut data = self.inner.borrow_mut();
            if let Some(ref mut grad) = data.gradient {
                grad.data.fill(1.0);
            }
        }

        let mut topo: Vec<Tensor> = Vec::new();
        let mut visited: HashSet<*const RefCell<TensorData>> = HashSet::new();

        fn build_topo(
            node: Tensor,
            topo: &mut Vec<Tensor>,
            visited: &mut HashSet<*const RefCell<TensorData>>,
        ) {
            let node_ptr = Rc::as_ptr(&node.inner);

            if !visited.contains(&node_ptr) {
                visited.insert(node_ptr);

                // Visit parents first
                if let Some(ref autograd_node) = node.inner.borrow().node {
                    for parent in &autograd_node.parents {
                        build_topo(parent.clone(), topo, visited);
                    }
                }

                topo.push(node.clone());
            }
        }

        build_topo(self.clone(), &mut topo, &mut visited);

        // Call backward on each node in reverse topological order
        for node in topo.iter().rev() {
            let autograd_node = node.inner.borrow().node.clone();

            if let Some(ref ag_node) = autograd_node {
                let lhs = ag_node.parents[0].clone();
                let rhs = ag_node.parents[1].clone();

                (ag_node.backwards)(lhs, rhs, node.clone());
            }
        }
    }
}
