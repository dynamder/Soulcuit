use std::borrow::Borrow;
use std::fmt::Debug;
use std::sync::Arc;
use super::basics::{FeedForwardFunc, HyperParamValue, HyperParams, InputMapFunc};
use nalgebra;
use tokio::sync::RwLock;

#[derive(Debug,Clone)]
pub struct Node {
    input_func: Arc<dyn InputMapFunc>,
    node_func: Arc<dyn FeedForwardFunc>,
    output_func: Arc<dyn FeedForwardFunc>,
    feedback_func: Arc<dyn FeedForwardFunc>,
    feedback_factor: Arc<RwLock<Vec<f32>>>,
    hyper_param: HyperParams
}
impl Node {
    pub fn new(
        input_func: impl InputMapFunc + 'static,
        node_func: impl FeedForwardFunc + 'static,
        output_func: impl FeedForwardFunc + 'static,
        feedback_func: impl FeedForwardFunc + 'static,
        feedback_len: usize,
        hyper_param: HyperParams
    ) -> Self {
        Node {
            input_func: Arc::new(input_func),
            node_func: Arc::new(node_func),
            output_func: Arc::new(output_func),
            feedback_func: Arc::new(feedback_func),
            feedback_factor: Arc::new(RwLock::new(vec![0.0; feedback_len])),
            hyper_param
        }
    }
    pub fn closure_new(
        input_func: Arc<dyn InputMapFunc>,
        node_func: Arc<dyn FeedForwardFunc>,
        output_func: Arc<dyn FeedForwardFunc>,
        feedback_func: Arc<dyn FeedForwardFunc>,
        feedback_len: usize,
        hyper_param: HyperParams
    ) -> Self {
        Node {
            input_func,
            node_func,
            output_func,
            feedback_func,
            feedback_factor: Arc::new(RwLock::new(vec![0.0; feedback_len])),
            hyper_param
        }
    }
    pub async fn process(&self, input: &[f32]) -> Vec<f32> {
        let mut feedback = self.feedback_factor.write().await;

        let mapped_input = self.input_func.forward(input, &*feedback, self.input_func.hyper_params());
        let node_forwarded = self.node_func.forward(&mapped_input, self.node_func.hyper_params());
        let output = self.output_func.forward(&node_forwarded, self.output_func.hyper_params());

        *feedback = self.feedback_func.forward(&output, self.feedback_func.hyper_params());
        output
    }
    pub async fn feedback(&self) -> Vec<f32> {
        self.feedback_factor.read().await.clone()
    }
    pub fn update_hyper_params(&mut self, params: HyperParams) {
        self.hyper_param = params;
    }
    pub fn add_hyper_param<T: Into<HyperParamValue>>(&mut self, name: &str, value: T) {
        self.hyper_param.set(name, value)
    }
}

