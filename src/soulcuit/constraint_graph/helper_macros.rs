#[macro_export]
macro_rules! make_forwardfunc {
    ($forward:expr, $hyper_params:expr) => {{
        std::sync::Arc::new($crate::soulcuit::constraint_graph::basics::ClosureFeedForwardFunc::new(
             $hyper_params,
             $forward,
        )) as std::sync::Arc<dyn $crate::soulcuit::constraint_graph::basics::FeedForwardFunc>
    }};

    ($forward:expr) => {{
        $crate::make_forwardfunc!($forward, $crate::soulcuit::constraint_graph::basics::HyperParams::new())
    }};
}
#[macro_export]
macro_rules! make_inputmapfunc {
    ($forward:expr, $hyper_params:expr) => {{
        std::sync::Arc::new($crate::soulcuit::constraint_graph::basics::ClosureInputMapFunc::new(
            $hyper_params,
            $forward,
        )) as std::sync::Arc<dyn $crate::soulcuit::constraint_graph::basics::InputMapFunc>
    }};

    ($forward:expr) => {{
        $crate::make_inputmapfunc!($forward, $crate::soulcuit::constraint_graph::basics::HyperParams::new())
    }};
}

#[macro_export]
macro_rules! make_node {
    (
        input_func: $input_func:expr,
        node_func: $node_func:expr,
        output_func: $output_func:expr,
        feedback_func: $feedback_func:expr,
        feedback_len: $feedback_len:expr
        $(, hyper_param: $hyper_param:expr)?
    ) => {{
        crate::soulcuit::constraint_graph::node::Node::closure_new(
            $input_func,
            $node_func,
            $output_func,
            $feedback_func,
            $feedback_len,
            crate::make_node!(@optional_hyper_param $($hyper_param)?)
        )
    }};

    (@optional_hyper_param $hyper_param:expr) => {
        $hyper_param
    };

    (@optional_hyper_param) => {
        $crate::soulcuit::constraint_graph::basics::HyperParams::new()
    };
}
#[macro_export]
macro_rules! make_port {
    ($port_func:expr, $hyper_params:expr) => {{
        std::sync::Arc::new(crate::soulcuit::constraint_graph::port::ClosureLayerPort::new (
            $hyper_params,
            $port_func
        )) as std::sync::Arc<dyn crate::soulcuit::constraint_graph::port::LayerPort>
    }};
    ($port_func:expr) => {{
        crate::make_port!($port_func, crate::soulcuit::constraint_graph::basics::HyperParams::new())
    }};
}