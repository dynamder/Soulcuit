extern crate proc_macro;
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, Data, DeriveInput, Fields, Ident};

#[proc_macro_derive(Vectorize, attributes(vector))]
pub fn vector_derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = input.ident;
    let fields = if let Data::Struct(data) = input.data {
        if let Fields::Named(fields) = data.fields {
            fields.named
        } else {
            panic!("Vectorize only supports structs with named fields");
        }
    } else {
        panic!("Vectorize only supports structs");
    };
    let field_names: Vec<&Ident> = fields.iter().map(|f| f.ident.as_ref().unwrap()).collect();
    let field_count = field_names.len();
    // 生成 From<Vec<f32>> 实现
    let from_impl = {
        let field_assignments = field_names.iter().enumerate().map(|(i, name)| {
            quote! { #name: v[#i] }
        });

        quote! {
            impl From<Vec<f32>> for #name {
                fn from(v: Vec<f32>) -> Self {
                    if v.len() != #field_count {
                        panic!("Invalid vector length for {}: expected {}, got {}",
                               stringify!(#name), #field_count, v.len());
                    }
                    Self { #(#field_assignments),* }
                }
            }
        }
    };
    // 生成 Vector trait 实现
    let vector_impl = {
        let len_impl = quote! { #field_count };

        let get_match = fields.iter().enumerate().map(|(i, name)| {
            quote! { #i => self.#name }
        });

        let to_fvec = fields.iter().map(|name| {
            quote! { self.#name }
        });

        quote! {
            impl Vector for #name {
                fn len(&self) -> usize {
                    #len_impl
                }

                fn get(&self, index: usize) -> f32 {
                    match index {
                        #(#get_match),*
                        _ => panic!("Invalid index: {}", index)
                    }
                }

                fn into_fvec(self) -> Vec<f32> {
                    vec![#(#to_fvec),*]
                }

                fn to_fvec(&self) -> Vec<f32> {
                    vec![#(self.#name),*]
                }
            }
        }
    };
    // 组合所有生成的代码
    let expanded = quote! {
        #from_impl
        #vector_impl
    };
    TokenStream::from(expanded)
}
