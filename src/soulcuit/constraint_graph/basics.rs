use std::collections::HashMap;
use std::fmt;
use std::fmt::Debug;
use dashmap::DashMap;

#[derive(Debug,Clone)]
pub enum HyperParamValue {
    Float(f32),
    Int(i32),
    Bool(bool),
    String(String),
    FVec(Vec<f32>),
    FMat(Vec<Vec<f32>>),
    IVec(Vec<i32>),
    IMat(Vec<Vec<i32>>),
}

impl HyperParamValue{
    pub fn as_float(&self) -> Option<f32> {
        match self { 
            HyperParamValue::Float(f) => Some(*f),
            _ => None,
        }
    }
    pub fn as_int(&self) -> Option<i32> {
        match self { 
            HyperParamValue::Int(i) => Some(*i),
            _ => None,
        }
    }
    pub fn as_bool(&self) -> Option<bool> {
        match self { 
            HyperParamValue::Bool(b) => Some(*b),
            _ => None,
        }
    }
    pub fn as_string(&self) -> Option<&String> {
        match self { 
            HyperParamValue::String(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_fvec(&self) -> Option<&Vec<f32>> {
        match self { 
            HyperParamValue::FVec(v) => Some(v),
            _ => None,
        }
    }
    pub fn as_fmat(&self) -> Option<&Vec<Vec<f32>>> {
        match self { 
            HyperParamValue::FMat(m) => Some(m),
            _ => None,
        }
    }
    pub fn as_ivec(&self) -> Option<&Vec<i32>> {
        match self { 
            HyperParamValue::IVec(v) => Some(v),
            _ => None,
        }
    }
    pub fn as_imat(&self) -> Option<&Vec<Vec<i32>>> {
        match self { 
            HyperParamValue::IMat(m) => Some(m),
            _ => None,
        }
    }
}
impl From<f32> for HyperParamValue {
    fn from(f: f32) -> Self {
        HyperParamValue::Float(f)
    }
}
impl From<i32> for HyperParamValue {
    fn from(i: i32) -> Self {
        HyperParamValue::Int(i)
    }
}
impl From<bool> for HyperParamValue {
    fn from(b: bool) -> Self {
        HyperParamValue::Bool(b)
    }
}
impl From<String> for HyperParamValue {
    fn from(s: String) -> Self {
        HyperParamValue::String(s)
    }
}
impl From<&str> for HyperParamValue {
    fn from(s: &str) -> Self {
        HyperParamValue::String(s.to_string())
    }
}
impl From<Vec<f32>> for HyperParamValue {
    fn from(v: Vec<f32>) -> Self {
        HyperParamValue::FVec(v)
    }
}
impl From<Vec<Vec<f32>>> for HyperParamValue {
    fn from(m: Vec<Vec<f32>>) -> Self {
        HyperParamValue::FMat(m)
    }
}
impl From<Vec<i32>> for HyperParamValue {
    fn from(v: Vec<i32>) -> Self {
        HyperParamValue::IVec(v)
    }
}
impl From<Vec<Vec<i32>>> for HyperParamValue {
    fn from(m: Vec<Vec<i32>>) -> Self {
        HyperParamValue::IMat(m)
    }
}
#[derive(Debug,Clone)]
pub struct HyperParams {
    params: HashMap<String, HyperParamValue>
}
impl HyperParams {
    pub fn new() -> Self {
        HyperParams {
            params: HashMap::new()
        }
    }
    pub fn set<T: Into<HyperParamValue>>(&mut self, name: &str, value: T) {
        self.params.insert(name.to_string(), value.into());
    }
    pub fn get(&self, name: &str) -> Option<&HyperParamValue> {
        self.params.get(name)
    }
    pub fn get_float(&self, name: &str) -> Option<f32> {
        self.params.get(name).and_then(|v| v.as_float())
    }
    pub fn get_int(&self, name: &str) -> Option<i32> {
        self.params.get(name).and_then(|v| v.as_int())
    }
    pub fn get_bool(&self, name: &str) -> Option<bool> {
        self.params.get(name).and_then(|v| v.as_bool())
    }
    pub fn get_string(&self, name: &str) -> Option<&String> {
        self.params.get(name).and_then(|v| v.as_string())
    }
    pub fn get_fvec(&self, name: &str) -> Option<&Vec<f32>> {
        self.params.get(name).and_then(|v| v.as_fvec())
    }
    pub fn get_fmat(&self, name: &str) -> Option<&Vec<Vec<f32>>> {
        self.params.get(name).and_then(|v| v.as_fmat())
    }
    pub fn get_ivec(&self, name: &str) -> Option<&Vec<i32>> {
        self.params.get(name).and_then(|v| v.as_ivec())
    }
    pub fn get_imat(&self, name: &str) -> Option<&Vec<Vec<i32>>> {
        self.params.get(name).and_then(|v| v.as_imat())
    }
    
}


pub trait InputMapFunc: Debug + Send + Sync {
    fn forward(&self, input: &[f32], feedback: &[f32], params: &HyperParams) -> Vec<f32>;
    fn hyper_params(&self) -> &HyperParams;
}
pub trait FeedForwardFunc: Debug + Send + Sync {
    fn forward(&self, input: &[f32], params: &HyperParams) -> Vec<f32>;
    fn hyper_params(&self) -> &HyperParams;
}

#[derive(Clone)]
pub struct ClosureFeedForwardFunc<F> {
    hyper_params: HyperParams,
    forward_func: F,
}
impl<F> ClosureFeedForwardFunc<F>
where
    F: Fn(&[f32], &HyperParams) -> Vec<f32> + Send + Sync + 'static,   
{
    pub fn new(hyper_params: HyperParams, forward_func: F) -> Self {
        ClosureFeedForwardFunc {
            hyper_params,
            forward_func,
        }
    }
}
impl<F> FeedForwardFunc for ClosureFeedForwardFunc<F>
where
    F: Fn(&[f32], &HyperParams) -> Vec<f32> + Send + Sync + 'static,
{
    fn forward(&self, input: &[f32], params: &HyperParams) -> Vec<f32> {
        (self.forward_func)(input, params)
    }

    fn hyper_params(&self) -> &HyperParams {
        &self.hyper_params
    }
}
impl<F> fmt::Debug for ClosureFeedForwardFunc<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FeedForwardFunc")
            .field("name", &"<closure>")
            .field("hyper_params", &self.hyper_params)
            .finish()
    }
}
// 用于包装闭包实现 InputMapFunc
#[derive(Clone)]
pub struct ClosureInputMapFunc<F> {
    hyper_params: HyperParams,
    forward_func: F,
}
impl<F> ClosureInputMapFunc<F>
where
    F: Fn(&[f32], &[f32], &HyperParams) -> Vec<f32> + Send + Sync + 'static,
{
    pub fn new(hyper_params: HyperParams, forward_func: F) -> Self {
        ClosureInputMapFunc {
            hyper_params,
            forward_func,
        }
    }
}
impl<F> InputMapFunc for ClosureInputMapFunc<F>
where
    F: Fn(&[f32], &[f32], &HyperParams) -> Vec<f32> + Send + Sync + 'static,
{
    fn forward(&self, input: &[f32], feedback: &[f32], params: &HyperParams) -> Vec<f32> {
        (self.forward_func)(input, feedback, params)
    }

    fn hyper_params(&self) -> &HyperParams {
        &self.hyper_params
    }
}
impl<F> fmt::Debug for ClosureInputMapFunc<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("InputMapFunc")
            .field("name", &"<closure>")
            .field("hyper_params", &self.hyper_params)
            .finish()
    }
}
// 宏定义





pub trait Vectorize: Debug + Send + Sync + From<Vec<f32>> {
    fn len(&self) -> usize;
    fn get(&self, index: usize) -> f32;
    
    fn into_fvec(self) -> Vec<f32>;
    
    fn to_fvec(&self) -> Vec<f32>;
    
}