//! The PDF object model (ISO 32000-2 section 7.3): the values a file is made of.

/// An indirect reference `num gen R`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ObjRef {
    /// The object number.
    pub num: u32,
    /// The generation number.
    pub generation: u16,
}

/// A dictionary: keys in the order the file wrote them; a repeated key keeps its first value.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Dict(Vec<(Vec<u8>, Object)>);

impl Dict {
    /// An empty dictionary.
    #[must_use]
    pub fn new() -> Self {
        Self(Vec::new())
    }

    /// The value under `key`, if any.
    #[must_use]
    pub fn get(&self, key: &[u8]) -> Option<&Object> {
        self.0.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    /// Add `key`, unless the dictionary already holds it.
    pub fn insert(&mut self, key: Vec<u8>, value: Object) {
        if self.get(&key).is_none() {
            self.0.push((key, value));
        }
    }

    /// Every key and value, in file order.
    pub fn iter(&self) -> impl Iterator<Item = (&[u8], &Object)> {
        self.0.iter().map(|(k, v)| (k.as_slice(), v))
    }

    /// Every value, to change in place.
    pub fn values_mut(&mut self) -> impl Iterator<Item = &mut Object> {
        self.0.iter_mut().map(|(_, v)| v)
    }

    /// How many entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether it has no entry.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

/// A stream: its dictionary and its data as stored in the file (still encoded, still encrypted).
#[derive(Debug, Clone, PartialEq)]
pub struct Stream {
    /// The stream dictionary.
    pub dict: Dict,
    /// The bytes between `stream` and `endstream`.
    pub raw: Vec<u8>,
}

/// A PDF value.
#[derive(Debug, Clone, PartialEq)]
pub enum Object {
    /// `null`.
    Null,
    /// `true` / `false`.
    Bool(bool),
    /// An integer.
    Integer(i64),
    /// A real number.
    Real(f64),
    /// A string's bytes (literal or hex; the encoding is the reader's to know).
    String(Vec<u8>),
    /// A name's bytes, `#xx` escapes resolved, without the `/`.
    Name(Vec<u8>),
    /// An array.
    Array(Vec<Object>),
    /// A dictionary.
    Dict(Dict),
    /// A stream (only ever an indirect object's value).
    Stream(Stream),
    /// An indirect reference.
    Reference(ObjRef),
}

impl Object {
    /// The integer, if this is one.
    #[must_use]
    pub fn as_integer(&self) -> Option<i64> {
        match self {
            Self::Integer(i) => Some(*i),
            _ => None,
        }
    }

    /// The number, integer or real.
    #[must_use]
    pub fn as_number(&self) -> Option<f64> {
        match self {
            #[expect(
                clippy::cast_precision_loss,
                reason = "an integer past 2^53 is a real to every PDF reader"
            )]
            Self::Integer(i) => Some(*i as f64),
            Self::Real(r) => Some(*r),
            _ => None,
        }
    }

    /// The name's bytes, if this is a name.
    #[must_use]
    pub fn as_name(&self) -> Option<&[u8]> {
        match self {
            Self::Name(n) => Some(n),
            _ => None,
        }
    }

    /// The dictionary, if this is one (a stream's dictionary too).
    #[must_use]
    pub fn as_dict(&self) -> Option<&Dict> {
        match self {
            Self::Dict(d) => Some(d),
            Self::Stream(s) => Some(&s.dict),
            _ => None,
        }
    }

    /// The array, if this is one.
    #[must_use]
    pub fn as_array(&self) -> Option<&[Object]> {
        match self {
            Self::Array(a) => Some(a),
            _ => None,
        }
    }

    /// The reference, if this is one.
    #[must_use]
    pub fn as_reference(&self) -> Option<ObjRef> {
        match self {
            Self::Reference(r) => Some(*r),
            _ => None,
        }
    }

    /// The kind of value, for an error message.
    #[must_use]
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Null => "null",
            Self::Bool(_) => "boolean",
            Self::Integer(_) => "integer",
            Self::Real(_) => "real",
            Self::String(_) => "string",
            Self::Name(_) => "name",
            Self::Array(_) => "array",
            Self::Dict(_) => "dictionary",
            Self::Stream(_) => "stream",
            Self::Reference(_) => "reference",
        }
    }
}
