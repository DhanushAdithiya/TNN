#![feature(portable_simd)]
pub mod raw_tensor;
pub mod tensor;

#[cfg(test)]
mod tensor_tests;
mod autodiff_test;
