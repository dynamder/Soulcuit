use std::fmt;
use std::fmt::Debug;
use std::sync::Arc;
use crate::soulcuit::constraint_graph::basics::HyperParams;

pub trait LayerPort: Debug + Send + Sync{
    fn process(&mut self, input: &[f32], hyper_params: &HyperParams) -> Vec<f32>;
    fn hyper_params(&self) -> &HyperParams;
}

#[derive(Clone)]
pub struct ClosureLayerPort<F> {
    hyper_params: HyperParams,
    port_func: F,
}
impl<F> ClosureLayerPort<F>
where
    F: Fn(&[f32], &HyperParams) -> Vec<f32> + Send + Sync + 'static,
{
    pub fn new(hyper_params: HyperParams, port_func: F) -> Self {
        ClosureLayerPort {
            hyper_params,
            port_func,
        }
    }
}
impl<F> LayerPort for ClosureLayerPort<F>
where
    F: Fn(&[f32], &HyperParams) -> Vec<f32> + Send + Sync + 'static,
{
    fn process(&mut self, input: &[f32], hyper_params: &HyperParams) -> Vec<f32> {
        (self.port_func)(input, hyper_params)
    }
    fn hyper_params(&self) -> &HyperParams {
        &self.hyper_params
    }
}
impl<F> fmt::Debug for ClosureLayerPort<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LayerPort")
            .field("name", &"<closure>")
            .field("hyper_params", &self.hyper_params)
            .finish()
    }
}



