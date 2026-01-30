use crate::raw_tensor::RawTensor;

type BackwardFn = fn(AutogradNode, AutogradNode, AutogradNode);

pub struct Tensor {
    tensor: RawTensor,
    gradient: Option<RawTensor>,
    node: Option<AutogradNode>
}

pub struct AutogradNode {
    parents: Vec<Tensor>,
    backwards: BackwardFn
}