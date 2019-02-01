use crate::{
    attr::{Attr, AttrClass, AttrField},
    bytes::Bytes,
    fixed::{Fixed, MutFixed},
    token::Token,
    variant::Variant,
};
use std::{
    fmt,
    marker::PhantomData,
    ops::{Deref, DerefMut, Range},
    slice,
};

#[repr(C)]
pub struct LayerStackData {
    pub children: Vec<MutFixed<Layer>>,
}

/// A mutable proxy for a layer object.
#[repr(C)]
pub struct LayerStack<'a> {
    data: *mut LayerStackData,
    depth: u8,
    add_child: extern "C" fn(*mut LayerStackData, *mut Layer),
    children_len: extern "C" fn(*const LayerStackData) -> u64,
    children_data: extern "C" fn(*const LayerStackData) -> *const MutFixed<Layer>,
    layer: *mut Layer,
    phantom: PhantomData<&'a ()>,
}

impl<'a> LayerStack<'a> {
    pub fn from_mut_ref(stack: &'a mut LayerStackData, layer: &'a mut Layer) -> LayerStack<'a> {
        LayerStack {
            data: stack,
            depth: 0,
            add_child: abi_add_child,
            children_len: abi_children_len,
            children_data: abi_children_data,
            layer,
            phantom: PhantomData,
        }
    }

    /// Returns the ID of self.
    pub fn id(&self) -> Token {
        self.deref().id()
    }

    /// Returns the type of self.
    pub fn data(&self) -> Bytes {
        self.deref().data()
    }

    /// Returns the slice of attributes.
    pub fn attrs(&self) -> Vec<Attr> {
        self.deref().attrs()
    }

    /// Find the attribute in the Layer.
    pub fn attr<T: Into<Token>>(&self, id: T) -> Option<Attr> {
        self.deref().attr(id)
    }

    /// Adds an attribute to the Layer.
    pub fn add_attr<C: AsRef<[Fixed<AttrClass>]>>(&mut self, attrs: &C, byte_range: Range<usize>) {
        self.deref_mut().add_attr(attrs, byte_range);
    }

    /// Returns the payload.
    pub fn payload(&self) -> Bytes {
        self.deref().payload()
    }

    /// Adds a payload to the Layer.
    pub fn set_payload(&mut self, payload: &Bytes) {
        self.deref_mut().set_payload(payload);
    }

    pub fn add_child<T: Into<MutFixed<Layer>>>(&mut self, layer: T) {
        (self.add_child)(self.data, layer.into().as_mut_ptr());
    }

    pub fn top(&self) -> Option<&Layer> {
        self.children().iter().rev().next().map(Deref::deref)
    }

    pub fn bottom(&self) -> Option<&Layer> {
        self.children().iter().next().map(Deref::deref)
    }

    fn children(&self) -> &[MutFixed<Layer>] {
        let data = (self.children_data)(self.data);
        let len = (self.children_len)(self.data) as usize;
        unsafe { slice::from_raw_parts(data, len) }
    }
}

impl<'a> Deref for LayerStack<'a> {
    type Target = Layer;

    fn deref(&self) -> &Layer {
        unsafe { &*self.layer }
    }
}

impl<'a> DerefMut for LayerStack<'a> {
    fn deref_mut(&mut self) -> &mut Layer {
        unsafe { &mut *self.layer }
    }
}

extern "C" fn abi_add_child(data: *mut LayerStackData, child: *mut Layer) {
    unsafe { (*data).children.push(MutFixed::from_ptr(child)) }
}

extern "C" fn abi_children_len(data: *const LayerStackData) -> u64 {
    unsafe { (*data).children.len() as u64 }
}

extern "C" fn abi_children_data(data: *const LayerStackData) -> *const MutFixed<Layer> {
    unsafe { (*data).children.as_ptr() }
}

#[repr(C)]
struct BoundAttr {
    attr: Fixed<AttrClass>,
    bit_range: Range<usize>,
}

/// A layer object.
#[repr(C)]
pub struct Layer {
    class: Fixed<LayerClass>,
    data: Bytes,
    attrs: Vec<BoundAttr>,
    payload: Bytes,
}

unsafe impl Send for Layer {}

impl Layer {
    /// Creates a new Layer.
    pub fn new<C: AsRef<Fixed<LayerClass>>>(class: &C, data: &Bytes) -> Layer {
        Layer {
            class: *class.as_ref(),
            data: *data,
            attrs: Vec::new(),
            payload: Bytes::new(),
        }
    }

    /// Returns the ID of self.
    pub fn id(&self) -> Token {
        self.class.id()
    }

    /// Returns the type of self.
    pub fn data(&self) -> Bytes {
        self.class.data(self)
    }

    /// Returns the slice of attributes.
    pub fn attrs(&self) -> Vec<Attr> {
        self.class.attrs(self)
    }

    /// Find the attribute in the Layer.
    pub fn attr<T: Into<Token>>(&self, id: T) -> Option<Attr> {
        let id = id.into();

        let headers = self
            .class
            .headers
            .iter()
            .map(|h| Attr::new(&h, h.bit_range(), self.data()));
        let attrs = self
            .attrs
            .iter()
            .map(|b| Attr::new(&b.attr, b.bit_range.clone(), self.data()));

        headers.chain(attrs).find(|attr| attr.is_match(id))
    }

    /// Adds an attribute to the Layer.
    pub fn add_attr<C: AsRef<[Fixed<AttrClass>]>>(&mut self, attrs: &C, byte_range: Range<usize>) {
        let func = self.class.add_attr;
        let attrs = attrs.as_ref();

        let root = attrs[0];
        let offset = (byte_range.start * 8) as isize - root.bit_range().start as isize;

        (func)(
            self,
            BoundAttr {
                attr: root,
                bit_range: (byte_range.start * 8)..(byte_range.end * 8),
            },
        );

        for attr in &attrs[1..] {
            let range = attr.bit_range();
            (func)(
                self,
                BoundAttr {
                    attr: *attr,
                    bit_range: (range.start as isize + offset) as usize
                        ..(range.end as isize + offset) as usize,
                },
            );
        }
    }

    /// Returns the payload.
    pub fn payload(&self) -> Bytes {
        self.class.payload(self)
    }

    /// Sets a payload to the Layer.
    pub fn set_payload(&mut self, payload: &Bytes) {
        let func = self.class.set_payload;
        (func)(self, payload.as_ptr(), payload.len() as u64);
    }
}

impl fmt::Debug for Layer {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Layer {:?}", self.id())
    }
}

impl Into<MutFixed<Layer>> for Layer {
    fn into(self) -> MutFixed<Layer> {
        MutFixed::new(self)
    }
}

/// A payload object.
#[repr(C)]
pub struct Payload {
    data: *const u8,
    len: u64,
    id: Token,
    typ: Token,
}

/// A builder object for LayerClass.
pub struct LayerClassBuilder {
    headers: Vec<Fixed<AttrClass>>,
}

impl LayerClassBuilder {
    /// Builds a new LayerClass.
    pub fn build(self) -> LayerClass {
        LayerClass {
            get_id: abi_id,
            data: abi_data,
            attrs_len: abi_attrs_len,
            attrs_data: abi_attrs_data,
            add_attr: abi_add_attr,
            set_payload: abi_set_payload,
            payload: abi_payload,
            headers: self.headers,
        }
    }
}

pub struct LayerType<T> {
    field: T,
    layer: Fixed<LayerClass>,
}

impl<T: fmt::Debug> fmt::Debug for LayerType<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("LayerType")
            .field("field", &self.field)
            .field("layer", &self.layer)
            .finish()
    }
}

impl<T> Deref for LayerType<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.field
    }
}

impl<T> DerefMut for LayerType<T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.field
    }
}

impl<T> AsRef<Fixed<LayerClass>> for LayerType<T> {
    fn as_ref(&self) -> &Fixed<LayerClass> {
        &self.layer
    }
}

impl<T: AttrField> LayerType<T> {
    pub fn new<D: Into<Token>>(id: D) -> Self {
        let mut ctx = T::context();
        ctx.id = id.into().to_string();
        let class = vec![AttrClass::builder(ctx.id.clone())
            .cast(|_, _| Ok(Variant::Bool(true)))
            .bit_range(ctx.bit_offset..(ctx.bit_offset + ctx.bit_size))
            .typ("@layer")];
        let class = class
            .into_iter()
            .chain(T::class(&ctx).into_iter())
            .map(|attr| Fixed::new(attr.build()))
            .collect();
        let layer = Fixed::new(LayerClass {
            get_id: abi_id,
            data: abi_data,
            attrs_len: abi_attrs_len,
            attrs_data: abi_attrs_data,
            add_attr: abi_add_attr,
            set_payload: abi_set_payload,
            payload: abi_payload,
            headers: class,
        });
        let field = T::new(&ctx);
        Self { field, layer }
    }

    pub fn byte_size(&self) -> usize {
        let range = self.layer.headers[0].byte_range();
        range.end - range.start
    }
}

/// A layer class object.
#[repr(C)]
pub struct LayerClass {
    get_id: extern "C" fn(*const LayerClass) -> Token,
    data: extern "C" fn(*const Layer, *mut u64) -> *const u8,
    attrs_len: extern "C" fn(*const Layer) -> u64,
    attrs_data: extern "C" fn(*const Layer) -> *const BoundAttr,
    add_attr: extern "C" fn(*mut Layer, BoundAttr),
    set_payload: extern "C" fn(*mut Layer, *const u8, u64),
    payload: extern "C" fn(*const Layer, *mut u64) -> *const u8,
    headers: Vec<Fixed<AttrClass>>,
}

impl LayerClass {
    /// Creates a new builder object for LayerClass.
    pub fn builder<H: Into<Vec<Fixed<AttrClass>>>>(headers: H) -> LayerClassBuilder {
        LayerClassBuilder {
            headers: headers.into(),
        }
    }

    fn id(&self) -> Token {
        (self.get_id)(self)
    }

    fn data(&self, layer: &Layer) -> Bytes {
        let mut len = 0;
        let data = (self.data)(layer, &mut len);
        unsafe { Bytes::from_raw_parts(data, len as usize) }
    }

    fn attrs(&self, layer: &Layer) -> Vec<Attr> {
        let data = (self.attrs_data)(layer);
        let len = (self.attrs_len)(layer) as usize;
        let headers = self
            .headers
            .iter()
            .map(|h| Attr::new(&h, h.bit_range(), layer.data()));
        let attrs = unsafe { slice::from_raw_parts(data, len) }
            .iter()
            .map(|b| Attr::new(&b.attr, b.bit_range.clone(), layer.data()));
        headers.chain(attrs).collect()
    }

    fn payload(&self, layer: &Layer) -> Bytes {
        let mut len = 0;
        let data = (self.payload)(layer, &mut len);
        unsafe { Bytes::from_raw_parts(data, len as usize) }
    }
}

impl fmt::Debug for LayerClass {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("LayerClass")
            .field("headers", &self.headers)
            .finish()
    }
}

impl Into<Fixed<LayerClass>> for &'static LayerClass {
    fn into(self) -> Fixed<LayerClass> {
        Fixed::from_static(self)
    }
}

extern "C" fn abi_id(class: *const LayerClass) -> Token {
    unsafe { (*class).headers[0].id() }
}

extern "C" fn abi_data(layer: *const Layer, len: *mut u64) -> *const u8 {
    unsafe {
        let data = &(*layer).data;
        *len = data.len() as u64;
        data.as_ptr()
    }
}

extern "C" fn abi_attrs_len(layer: *const Layer) -> u64 {
    unsafe { (*layer).attrs.len() as u64 }
}

extern "C" fn abi_attrs_data(layer: *const Layer) -> *const BoundAttr {
    unsafe { (*layer).attrs.as_ptr() }
}

extern "C" fn abi_add_attr(layer: *mut Layer, attr: BoundAttr) {
    let attrs = unsafe { &mut (*layer).attrs };
    attrs.push(attr);
}

extern "C" fn abi_payload(layer: *const Layer, len: *mut u64) -> *const u8 {
    unsafe {
        let data = &(*layer).payload;
        *len = data.len() as u64;
        data.as_ptr()
    }
}

extern "C" fn abi_set_payload(layer: *mut Layer, data: *const u8, len: u64) {
    unsafe {
        let layer = &mut (*layer);
        layer.payload = Bytes::from_raw_parts(data, len as usize);
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        attr::AttrClass,
        bytes::Bytes,
        fixed::Fixed,
        layer::{Layer, LayerClass},
        token::Token,
        variant::Variant,
    };

    #[test]
    fn id() {
        let id = Token::from(123);
        let attr = vec![Fixed::new(AttrClass::builder(id).build())];
        let class = Box::new(Fixed::new(LayerClass::builder(attr).build()));
        let layer = Layer::new(&class, &Bytes::new());
        assert_eq!(layer.id(), id);
    }

    #[test]
    fn data() {
        let data = b"hello";
        let attr = vec![Fixed::new(AttrClass::builder(Token::null()).build())];
        let class = Box::new(Fixed::new(LayerClass::builder(attr).build()));
        let layer = Layer::new(&class, &Bytes::from(&data[..]));
        assert_eq!(layer.data(), Bytes::from(&data[..]));
    }

    #[test]
    fn attrs() {
        let attr = vec![Fixed::new(AttrClass::builder(Token::null()).build())];
        let class = Box::new(Fixed::new(LayerClass::builder(attr).build()));
        let mut layer = Layer::new(&class, &Bytes::new());

        struct Class(Vec<Fixed<AttrClass>>);
        impl AsRef<[Fixed<AttrClass>]> for Class {
            fn as_ref(&self) -> &[Fixed<AttrClass>] {
                &self.0
            }
        }

        let class = Class(vec![Fixed::new(
            AttrClass::builder("nil")
                .typ("@nil")
                .cast(|_, _| Ok(Variant::Nil))
                .build(),
        )]);

        let count = 100;
        for i in 0..count {
            layer.add_attr(&class, 0..i);
        }
        let mut iter = layer.attrs().into_iter();
        iter.next();
        for i in 0..count {
            let attr = iter.next().unwrap();
            assert_eq!(attr.id(), Token::from("nil"));
            assert_eq!(attr.typ(), Token::from("@nil"));
            assert_eq!(attr.byte_range(), 0..i);
        }
        assert!(iter.next().is_none());
    }
}
