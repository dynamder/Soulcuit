use std::cmp::Ordering;
use std::fmt::{Debug, Formatter};
use itertools::Itertools;
use super::basics::{FeedForwardFunc, InputMapFunc, HyperParams};
#[derive(Debug)]
pub struct ReLU {
    hyper_param: HyperParams
}
impl ReLU {
    pub fn new() -> Self {
        ReLU {
            hyper_param: HyperParams::new()
        }
    }
}


impl FeedForwardFunc for ReLU {
    fn forward(&self, input: &[f32], _hyper_param: &HyperParams) -> Vec<f32> {
        input.iter().map(|x| x.max(0.0)).collect()
    }
    fn hyper_params(&self) -> &HyperParams { //always Empty
        &self.hyper_param
    }
}

#[derive(Debug)]
pub struct LeakyReLU {
    hyper_param: HyperParams
}
impl LeakyReLU {
    pub fn new(alpha: f32) -> Self {
        let mut param = HyperParams::new();
        param.set("alpha", alpha);
        LeakyReLU {
            hyper_param: param
        }
    }
}
impl FeedForwardFunc for LeakyReLU {
    fn forward(&self, input: &[f32], _: &HyperParams) -> Vec<f32> {
        input.iter().map(|x| x.max(self.hyper_param.get_float("alpha").unwrap_or(0.0) * x)).collect()
    }
    fn hyper_params(&self) -> &HyperParams {
        &self.hyper_param
    }
}

#[derive(Debug)]
pub struct Tanh {
    hyper_param: HyperParams
}
impl Tanh {
    pub fn new() -> Self {
        Tanh {
            hyper_param: HyperParams::new()
        }
    }
}
impl FeedForwardFunc for Tanh {
    fn forward(&self, input: &[f32], _: &HyperParams) -> Vec<f32> {
        input.iter().map(|x| x.tanh()).collect()
    }
    fn hyper_params(&self) -> &HyperParams { //always empty
        &self.hyper_param
    }
}

#[derive(Debug)]
pub struct Sigmoid {
    hyper_param: HyperParams
}
impl Sigmoid {
    pub fn new() -> Self {
        Sigmoid {
            hyper_param: HyperParams::new()
        }
    }
}
impl FeedForwardFunc for Sigmoid {
    fn forward(&self, input: &[f32], _: &HyperParams) -> Vec<f32> {
        input.iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect()
    }
    fn hyper_params(&self) -> &HyperParams { //always empty
        &self.hyper_param
    }
}

#[derive(Debug)]
pub struct Softmax {
    hyper_param: HyperParams
}
impl Softmax {
    pub fn new() -> Self {
        Softmax {
            hyper_param: HyperParams::new()
        }
    }
}
impl FeedForwardFunc for Softmax {
    fn forward(&self, input: &[f32], _: &HyperParams) -> Vec<f32> {
        let sum = input.iter().map(|x| x.exp()).sum::<f32>();
        input.iter().map(|x| x.exp() / sum).collect()
    }
    fn hyper_params(&self) -> &HyperParams { //always empty
        &self.hyper_param
    }
}
#[derive(Debug)]
pub struct Sparsemax {
    hyper_param: HyperParams
}
impl Sparsemax {
    pub fn new() -> Self {
        Sparsemax {
            hyper_param: HyperParams::new()
        }
    }
}
impl FeedForwardFunc for Sparsemax { //TODO: need test
    fn forward(&self, input: &[f32], _: &HyperParams) -> Vec<f32> {
        // 1. 创建排序索引（无需实际排序向量）
        let mut indices: Vec<usize> = (0..input.len()).collect();
        indices.sort_unstable_by(|&a, &b| input[b].partial_cmp(&input[a]).unwrap());
        // 2. 计算累积和并找到支持集大小
        let (k, cumsum) = indices.iter()
            .enumerate()
            .scan(0.0, |acc, (i, &idx)| {
                *acc += input[idx];
                Some((i, *acc))
            })
            .take_while(|&(i, sum)| {
                let condition = 1.0 + (i + 1) as f32 * input[indices[i]];
                condition > sum
            })
            .last()
            .map_or((0, 0.0), |(i, sum)| (i + 1, sum));
        // 3. 计算阈值 τ(z)
        let tau = if k > 0 { (cumsum - 1.0) / k as f32 } else { f32::NEG_INFINITY };
        // 4. 应用稀疏化转换
        input.iter()
            .map(|&x| (x - tau).max(0.0))
            .collect()
    }
    fn hyper_params(&self) -> &HyperParams {
        &self.hyper_param
    }
}

#[derive(Debug)]
pub struct Swish {
    hyper_param: HyperParams
}
impl Swish {
    pub fn new(beta: f32) -> Self {
        let mut param = HyperParams::new();
        param.set("beta", beta);
        Swish {
            hyper_param: param
        }
    }
}
impl FeedForwardFunc for Swish {
    fn forward(&self, input: &[f32], _: &HyperParams) -> Vec<f32> {
        input.iter().map(|x| x / (1.0 + (-self.hyper_param.get_float("beta").unwrap_or(1.0) * x).exp())).collect()
    }
    fn hyper_params(&self) -> &HyperParams {
        &self.hyper_param
    }
}

#[derive(Debug)]
pub struct Softplus {
    hyper_param: HyperParams
}
impl Softplus {
    pub fn new() -> Self {
        Softplus {
            hyper_param: HyperParams::new()
        }
    }
}
impl FeedForwardFunc for Softplus {
    fn forward(&self, input: &[f32], _: &HyperParams) -> Vec<f32> {
        input.iter().map(|x| (1.0 + x.exp()).ln()).collect()
    }
    fn hyper_params(&self) -> &HyperParams { //always empty
        &self.hyper_param
    }
}
#[derive(Debug)]
pub struct Softsign {
    hyper_param: HyperParams
}
impl Softsign {
    pub fn new() -> Self {
        Softsign {
            hyper_param: HyperParams::new()
        }
    }
}
impl FeedForwardFunc for Softsign {
    fn forward(&self, input: &[f32], _: &HyperParams) -> Vec<f32> {
        input.iter().map(|x| x / (1.0 + x.abs())).collect()
    }
    fn hyper_params(&self) -> &HyperParams { //always empty
        &self.hyper_param
    }
}

#[derive(Debug)]
pub struct Linear {
    hyper_param: HyperParams
}
impl Linear {
    pub fn new() -> Self {
        Linear {
            hyper_param: HyperParams::new()
        }
    }
}
impl FeedForwardFunc for Linear {
    fn forward(&self, input: &[f32], _: &HyperParams) -> Vec<f32> {
        Vec::from(input)
    }
    fn hyper_params(&self) -> &HyperParams { //always empty
        &self.hyper_param
    }
}

#[derive(Debug)]
pub struct Dramaend {
    hyper_param: HyperParams
}
impl Dramaend {
    pub fn new(a_pos: f32, a_neg: f32, b: f32) -> Self {
        let mut param = HyperParams::new();
        param.set("a_pos", a_pos);
        param.set("a_neg", a_neg);
        param.set("b", b);
        Dramaend {
            hyper_param: param
        }
    }
}
impl FeedForwardFunc for Dramaend {
    fn forward(&self, input: &[f32], params: &HyperParams) -> Vec<f32> {
        let a_pos = params.get_float("a_pos").unwrap_or(1.0);
        let a_neg = params.get_float("a_neg").unwrap_or(1.0);
        let b = params.get_float("b").unwrap_or(1.0);
        input.iter()
            .map(|x| {
                if *x >=0.0 {
                    b/a_pos * ((a_pos*(*x).exp())-1.0)
                }else{
                    -b/a_neg * ((-a_neg*(*x).exp())-1.0)
                }
            }).collect()
    }
    fn hyper_params(&self) -> &HyperParams {
        &self.hyper_param
    }
}

#[derive(Debug)]
pub struct NatureDecay {
    hyper_param: HyperParams,
    last_called: f32,
}
impl NatureDecay { 
    pub fn new(tau: f32) -> Self {
        let mut param = HyperParams::new();
        param.set("tau", tau);
        NatureDecay {
            hyper_param: param,
            last_called: (chrono::Local::now().timestamp_millis() / 1000) as f32,
        }
    }
}

impl FeedForwardFunc for NatureDecay {
    fn forward(&self, input: &[f32], params: &HyperParams) -> Vec<f32> {
        let tau = params.get_float("tau").unwrap();
        let time_undergo = (chrono::Local::now().timestamp_millis() / 1000) as f32 - self.last_called;
        input.iter().map(|x| x*(-time_undergo/tau).exp()).collect()
    }
    fn hyper_params(&self) -> &HyperParams {
        &self.hyper_param
    }
}

#[derive(Debug)]
pub struct MockDecay {
    hyper_param: HyperParams,
    last_called: f32,
}
impl MockDecay {
    pub fn new(alpha: f32, gamma: f32, tau: f32) -> Self {
        let mut param = HyperParams::new();
        param.set("alpha", alpha);
        param.set("gamma", gamma);
        param.set("tau", tau);
        MockDecay {
            hyper_param: param,
            last_called: (chrono::Local::now().timestamp_millis() / 1000) as f32,
        }
    }
}

impl FeedForwardFunc for MockDecay {
    fn forward(&self, input: &[f32], params: &HyperParams) -> Vec<f32> {
        let alpha = params.get_float("alpha").unwrap();
        let gamma = params.get_float("gamma").unwrap();
        let tau = params.get_float("tau").unwrap();
        let time_undergo = (chrono::Local::now().timestamp_millis() / 1000) as f32 - self.last_called;
        input.iter().map(|x| x*(1.0-gamma)*alpha.powf(-time_undergo/tau)).collect()
    }
    fn hyper_params(&self) -> &HyperParams {
        &self.hyper_param
    }
}