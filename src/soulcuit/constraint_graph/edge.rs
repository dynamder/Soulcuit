use std::sync::Arc;
use super::basics::FeedForwardFunc;

#[derive(Debug,Clone)]
pub struct Edge {
    modulator: Arc<dyn FeedForwardFunc>
}
impl Edge {
    pub fn new(modulator: impl FeedForwardFunc + 'static) -> Self {
        Edge {
            modulator: Arc::new(modulator)
        }
    }
    pub fn modulate(&self, input: &[f32]) -> Vec<f32> {
        self.modulator.forward(input, self.modulator.hyper_params())
    }
}