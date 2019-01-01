use crate::{
    attr::{Attr, AttrClass, AttrClassBuilder, AttrContext, AttrField},
    slice,
    variant::Variant,
};
use std::{convert::Into, io::Result};

/// Cast trait.
pub trait Cast: 'static + Send + Sync + CastClone {
    fn cast(&self, attr: &Attr, data: &slice::ByteSlice) -> Result<Variant>;
}

pub trait CastClone {
    fn clone_box(&self) -> Box<Cast>;
}

impl<T> CastClone for T
where
    T: 'static + Cast + Clone,
{
    fn clone_box(&self) -> Box<Cast> {
        Box::new(self.clone())
    }
}

impl Clone for Box<Cast> {
    fn clone(&self) -> Box<Cast> {
        self.clone_box()
    }
}

/// Typed cast trait.
pub trait Typed: Cast {
    type Output: Into<Variant>;
    fn cast(&self, _attr: &Attr, data: &slice::ByteSlice) -> Result<Self::Output>;
    fn bit_size(&self) -> usize {
        8
    }
}

/// Mappable cast trait.
pub trait Map
where
    Self: Sized,
{
    fn map<I, R, F>(self, func: F) -> Mapped<Self, I, R, F>
    where
        Self: Typed<Output = I>,
        I: Into<Variant>,
        R: Into<Variant>,
        F: Fn(I) -> R + Clone,
    {
        Mapped { cast: self, func }
    }
}

impl<T, X> Map for T
where
    T: Typed<Output = X>,
    X: Into<Variant>,
{
}

#[derive(Clone)]
pub struct Mapped<T, I, R, F>
where
    T: Typed<Output = I>,
    I: Into<Variant>,
    R: Into<Variant>,
    F: Fn(I) -> R + Clone,
{
    cast: T,
    func: F,
}

impl<T, I, R, F> Typed for Mapped<T, I, R, F>
where
    T: Typed<Output = I> + Clone + 'static,
    I: Into<Variant> + Clone + 'static,
    R: Into<Variant> + Clone + 'static,
    F: Fn(I) -> R + Clone + Send + Sync + 'static,
{
    type Output = R;

    fn cast(&self, attr: &Attr, data: &slice::ByteSlice) -> Result<Self::Output> {
        Typed::cast(&self.cast, attr, data).map(self.func.clone())
    }
}

impl<T, X> Cast for T
where
    T: Typed<Output = X> + Clone,
    X: Into<Variant>,
{
    fn cast(&self, attr: &Attr, data: &slice::ByteSlice) -> Result<Variant> {
        Typed::cast(self, attr, data).map(|r| r.into())
    }
}

#[derive(Default, Clone)]
pub struct Nil();

impl Typed for Nil {
    type Output = Variant;

    fn cast(&self, _attr: &Attr, _data: &slice::ByteSlice) -> Result<Variant> {
        Ok(Variant::Nil)
    }
}

impl<I: 'static + Into<Variant> + Clone, T: Typed<Output = I> + Clone + Default> AttrField for T {
    type Builder = TypedBuilder<I, T>;

    fn from_builder(builder: &Self::Builder) -> Self {
        Self::default()
    }
}

pub struct TypedBuilder<I: 'static + Into<Variant> + Clone, T: Default + Typed<Output = I>> {
    data: T,
    pub path: String,
    pub typ: String,
    pub name: &'static str,
    pub desc: &'static str,
    pub aliases: Vec<String>,
    pub bit_offset: usize,
    pub bit_size: usize,
    mapper: fn(I) -> Variant,
}

impl<I: 'static + Into<Variant> + Clone, T: Typed<Output = I> + Clone + Default> Default
    for TypedBuilder<I, T>
{
    fn default() -> Self {
        let data = T::default();
        let bit_size = data.bit_size();
        Self {
            data,
            path: String::default(),
            typ: String::default(),
            name: "",
            desc: "",
            aliases: Vec::new(),
            bit_offset: 0,
            bit_size,
            mapper: |x| x.into(),
        }
    }
}

impl<I: 'static + Into<Variant> + Clone, T: Typed<Output = I> + Clone + Default>
    Into<AttrClassBuilder> for TypedBuilder<I, T>
{
    fn into(self) -> AttrClassBuilder {
        AttrClass::builder(&self.path)
            .typ(&self.typ)
            .cast(&self.data.map(self.mapper))
            .aliases(self.aliases)
            .bit_range(0, self.bit_offset..(self.bit_offset + self.bit_size))
            .name(self.name)
            .description(self.desc)
    }
}
