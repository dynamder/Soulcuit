use schemars::JsonSchema;
use vector_derive::Vectorize;
use serde::{Deserialize, Serialize};
use super::basics::Vectorize;
#[derive(Debug,Clone,Vectorize,Serialize,Deserialize,JsonSchema)]
pub struct EventVec {
    pub valence: f32,
    pub intensity: f32,
    pub novelty: f32,
    pub duration: f32,

    pub relevance: f32,
    pub congruence: f32,
    pub self_involve: f32,
    pub controllability: f32,
    pub attribution: f32,

    pub social: f32,
    pub public: f32,
    pub rel: f32,
    pub power: f32,
}


#[derive(Debug,Clone,Vectorize,Serialize,Deserialize,JsonSchema)]
pub struct TraitModEmo {
    pub neg_amplifier: f32,
    pub pos_regulator: f32,
    pub rage_generator: f32,
    pub guilt_amplifier: f32,
    pub novel_modulator: f32,
    pub social_modulator: f32,
    pub restore: f32,
}

#[derive(Debug,Clone,Vectorize,Serialize,Deserialize,JsonSchema)]
pub struct TraitModPhy {
    pub stress_buffer: f32,
    pub restore_accelerator: f32,
    pub cog_drain: f32,
    pub fatigue_modulator: f32,
    pub alertness_modulator: f32,
    pub cog_flex: f32,
    pub cog_bias: f32,
}
