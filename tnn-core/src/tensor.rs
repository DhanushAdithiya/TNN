#![allow(unused)]
use crate::raw_tensor::RawTensor;
use std::cell::RefCell;
use std::collections::HashSet;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::rc::Rc;
use std::collections::BTreeSet;

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
    println!("I AM HERE");
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

impl<'a> Add<&'a Tensor> for &'a Tensor {
    type Output = Tensor;

    fn add(self, rhs: &'a Tensor) -> Self::Output {
        let lhs_data = self.inner.borrow();
        let rhs_data = rhs.inner.borrow();
        
        let o = lhs_data.tensor.add(&rhs_data.tensor);
        let shape = lhs_data.tensor.shape();
        
        drop(rhs_data);

        let node = AutogradNode {
            parents: vec![self.clone(), rhs.clone()],  // Clone just increments Rc counter
            sign: Op::Add,
            backwards: backwards_add,
        };

        Tensor {
            inner: Rc::new(RefCell::new(TensorData {
                tensor: o,
                gradient: Some(RawTensor::zeros(shape)),
                node: Some(node),
            }))
        }
    }
}

impl<'a> Sub<&'a TensorData> for &'a TensorData {
    type Output = TensorData;

    fn sub(self, rhs: &'a TensorData) -> Self::Output {
        let o = self.tensor.sub(&rhs.tensor);
        let mut op = TensorData {
            tensor: o,
            gradient: Some(RawTensor::zeros(self.tensor.shape())),
            node: None
        };
        return op;
    }
}



impl<'a> Div<&'a TensorData> for &'a TensorData {
    type Output = TensorData;

    fn div(self, rhs: &'a TensorData) -> Self::Output {
        let o = self.tensor.div(&rhs.tensor);
        let mut op = TensorData {
            tensor: o,
            gradient: Some(RawTensor::zeros(self.tensor.shape())),
            node: None
        };
        return op;
    }
}

impl<'a> Mul<&'a TensorData> for &'a TensorData {
    type Output = TensorData;

    fn mul(self, rhs: &'a TensorData) -> Self::Output {
        let o = self.tensor.mul(&rhs.tensor);
        let mut op = TensorData {
            tensor: o,
            gradient: Some(RawTensor::zeros(self.tensor.shape())),
            node: None
        };
        return op;
    }
}


impl Neg for TensorData {
    type Output = Self;

    fn neg(mut self) -> Self::Output {
        return TensorData {
            tensor: self.tensor.scale(-1.0),
            gradient: self.gradient,
            node: self.node,
        };
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
            }))
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
            visited: &mut HashSet<*const RefCell<TensorData>>
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