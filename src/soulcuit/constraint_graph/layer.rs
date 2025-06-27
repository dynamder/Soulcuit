use std::sync::Arc;
use crate::{make_forwardfunc, make_inputmapfunc, make_node, make_port};
use crate::soulcuit::constraint_graph::basics::{FeedForwardFunc, HyperParams, InputMapFunc};
use crate::soulcuit::constraint_graph::func_collection::{MockDecay, NatureDecay};
use crate::soulcuit::constraint_graph::port::LayerPort;
use super::{node::Node, edge::Edge, func_collection};
use super::vector::*;


pub trait Layer {
    async fn forward(&self, input: &[f32]) -> Vec<f32>;
    async fn update(&mut self);

    async fn preprocess(&mut self, input: &[f32]) -> Vec<f32>;
    async fn postprocess(&mut self,input: &[f32]) -> Vec<f32>;
}

//TODO: let's get rid of the bullshit traits now and quickly build a prototype graph

pub struct EventLayer {
    event_history: Vec<EventVec>,
    node_list: Vec<Node>,
}
impl EventLayer {
    pub fn new(init_event_history: Vec<EventVec>) -> Self {
        EventLayer {
            event_history: init_event_history,
            node_list: vec![
                make_node!(
                    input_func: make_inputmapfunc!(
                        |input: &[f32], feedback: &[f32], params: &HyperParams| -> Vec<f32> {
                            Vec::from(input)
                        }),
                    node_func: make_forwardfunc!(
                        |input: &[f32], params: &HyperParams| -> Vec<f32> {
                            let sum = input.iter().map(|x| x.abs()).sum::<f32>();
                            input.iter().map(|x| x/sum).collect()
                        }),
                    output_func: make_forwardfunc!(
                        |input: &[f32], params: &HyperParams| -> Vec<f32> {
                            Vec::from(input)
                        }),
                    feedback_func: make_forwardfunc!(
                        |input: &[f32], params: &HyperParams| -> Vec<f32> {
                            vec![]
                        }),
                    feedback_len: 0
                )
            ]
        }
    }
}
impl Layer for EventLayer {
    async fn forward(&self, input: &[f32]) -> Vec<f32> {
        let node = self.node_list.first()
            .expect("EventLayer::forward: node_list is empty");
        node.process(input).await
    }
    async fn update(&mut self) {

    }
    async fn preprocess(&mut self, input: &[f32]) -> Vec<f32> { //decay and event fuse
        self.event_history.iter_mut()
            .for_each(|mut event| {
                let nature_decay = NatureDecay::new(event.duration);
                [event.valence,event.intensity,event.novelty] = *nature_decay.forward(
                    &vec![event.valence,event.intensity,event.novelty],nature_decay.hyper_params()
                );
                //TODO: adjust mock_decay hyper_param
                let mock_decay = MockDecay::new(0.2,1.5,event.duration);
                [event.rel,event.power] = *mock_decay.forward(
                    &vec![event.rel,event.power],mock_decay.hyper_params()
                );
            });
        let (fuse_valence,fuse_intensity,fuse_congruence,fuse_self_involve,fuse_controllability,fuse_public,fuse_rel,fuse_power) = 
            self.event_history.iter()
            .fold(
                (0.,0.,0.,0.,0.,0.,0.,0.),
                |(fuse_valence,fuse_intensity,fuse_congruence,fuse_self_involve,fuse_controllability,fuse_public,fuse_rel,fuse_power),event| {
                    (
                        fuse_valence + event.valence,
                        fuse_intensity + event.intensity,
                        fuse_congruence + event.congruence,
                    )
                }
            )
    }
    async fn postprocess(&mut self, input: &[f32]) -> Vec<f32> {
        todo!()
    }
}