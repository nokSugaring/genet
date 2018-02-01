#![allow(non_upper_case_globals)]

extern crate libc;

use std;
use std::ffi::CString;
use super::variant::Variant;
use super::layer::Layer;
use super::payload::Payload;
use super::attr::Attr;
use super::context::Context;
use super::token::Token;

macro_rules! def_func {
    ( $( $x:ident, $y:ty ); *; ) => {
        $(
        pub(crate) static mut $x: Option<$y> = None;
        )*
        pub fn init(resolve: fn(*const libc::c_char) -> *const ()) {
            unsafe {
                $(
                    $x = std::mem::transmute(
                        resolve(CString::new(stringify!($x)).unwrap().as_ptr()));
                    $x.expect(concat!("symbol not found: ", stringify!($x)));
                )*
            }
        }
    };
}

def_func!(
    Token_literal_,        extern "C" fn(*const libc::c_char, libc::size_t) -> Token;
    Token_string,          extern "C" fn(Token) -> *const libc::c_char;
    Token_join,            extern "C" fn(Token, Token) -> Token;
    Context_getOption,     extern "C" fn(*const Context, *const libc::c_char) -> *const Variant;
    Variant_setNil,        extern "C" fn(*mut Variant);
    Variant_setBool,       extern "C" fn(*mut Variant, bool);
    Variant_setInt64,      extern "C" fn(*mut Variant, i64);
    Variant_setUint64,     extern "C" fn(*mut Variant, u64);
    Variant_setDouble,     extern "C" fn(*mut Variant, f64);
    Variant_string,        extern "C" fn(*const Variant) -> *const libc::c_char;
    Variant_setString,     extern "C" fn(*mut Variant, *const libc::c_char);
    Variant_setSlice,      extern "C" fn(*mut Variant, (*const u8, *const u8));
    Variant_arrayValue,    extern "C" fn(*const Variant, libc::size_t) -> *const Variant;
    Variant_arrayValueRef, extern "C" fn(*mut Variant, libc::size_t) -> *mut Variant;
    Variant_mapValue,      extern "C" fn(*const Variant, *const libc::c_char) -> *const Variant;
    Variant_mapValueRef,   extern "C" fn(*mut Variant, *const libc::c_char) -> *mut Variant;
    Layer_attr,            extern "C" fn(*const Layer, Token) -> *const Attr;
    Layer_payloads,        extern "C" fn(*const Layer, *mut libc::size_t) -> *const *const Payload;
    Layer_addLayer,        extern "C" fn(*mut Layer, *mut Context, Token) -> *mut Layer;
    Layer_addSubLayer,     extern "C" fn(*mut Layer, *mut Context, Token) -> *mut Layer;
    Layer_addAttr,         extern "C" fn(*mut Layer, *mut Context, Token) -> *mut Attr;
    Layer_addPayload,      extern "C" fn(*mut Layer, *mut Context) -> *mut Payload;
    Layer_addTag,          extern "C" fn(*mut Layer, Token);
    Payload_addSlice,      extern "C" fn(*mut Payload, (*const u8, *const u8));
    Payload_slices,        extern "C" fn(*const Payload, *mut libc::size_t) -> *const (*const u8, *const u8);
    Payload_addAttr,       extern "C" fn(*mut Payload, *mut Context, Token) -> *mut Attr;
);

#[macro_export]
macro_rules! plugkit_module(
    ($body:expr) => (
        #[no_mangle]
        pub extern "C" fn plugkit_v1_init(resolve: fn(*const libc::c_char) -> *const ()) {
            plugkit::symbol::init(resolve);
            fn init() {
                $body
            }
            init();
        }
    )
);