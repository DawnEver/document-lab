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

/// A stream: its dictionary and its data as stored in the file, still encoded (decrypted once read from an encrypted
/// file).
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

/// A text string's text (7.9.2.2): UTF-16BE after its byte order mark, UTF-8 after its mark, else
/// `PDFDocEncoding` (D.2). A byte or a code unit no character stands for reads as U+FFFD.
#[must_use]
pub fn text_string(bytes: &[u8]) -> String {
    if let Some(utf16) = bytes.strip_prefix(&[0xFE, 0xFF]) {
        let units = utf16
            .chunks(2)
            .map(|pair| u16::from_be_bytes([pair[0], pair.get(1).copied().unwrap_or(0)]));
        return char::decode_utf16(units)
            .map(|c| c.unwrap_or(char::REPLACEMENT_CHARACTER))
            .collect();
    }
    if let Some(utf8) = bytes.strip_prefix(&[0xEF, 0xBB, 0xBF]) {
        return String::from_utf8_lossy(utf8).into_owned();
    }
    bytes.iter().map(|&b| pdf_doc_char(b)).collect()
}

/// The character `PDFDocEncoding` gives `byte`.
fn pdf_doc_char(byte: u8) -> char {
    /// 0x18..=0x1F: accents.
    const ACCENTS: [char; 8] = [
        '\u{2D8}', '\u{2C7}', '\u{2C6}', '\u{2D9}', '\u{2DD}', '\u{2DB}', '\u{2DA}', '\u{2DC}',
    ];
    /// 0x80..=0x9E: punctuation, ligatures and letters Latin-1 has not.
    const HIGH: [char; 31] = [
        '\u{2022}', '\u{2020}', '\u{2021}', '\u{2026}', '\u{2014}', '\u{2013}', '\u{192}',
        '\u{2044}', '\u{2039}', '\u{203A}', '\u{2212}', '\u{2030}', '\u{201E}', '\u{201C}',
        '\u{201D}', '\u{2018}', '\u{2019}', '\u{201A}', '\u{2122}', '\u{FB01}', '\u{FB02}',
        '\u{141}', '\u{152}', '\u{160}', '\u{178}', '\u{17D}', '\u{131}', '\u{142}', '\u{153}',
        '\u{161}', '\u{17E}',
    ];
    match byte {
        0x18..=0x1F => ACCENTS[usize::from(byte - 0x18)],
        0x80..=0x9E => HIGH[usize::from(byte - 0x80)],
        0xA0 => '\u{20AC}',
        0x7F | 0x9F | 0xAD => char::REPLACEMENT_CHARACTER,
        _ => char::from(byte),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_text_string_reads_by_its_byte_order_mark() {
        assert_eq!(
            text_string(b"\xFE\xFF\x00\xE9\xD8\x3D\xDE\x00"),
            "\u{e9}\u{1F600}"
        );
        assert_eq!(text_string(b"\xEF\xBB\xBF\xC3\xA9"), "\u{e9}");
        assert_eq!(
            text_string(b"a\x93\xA0\xE9\x18"),
            "a\u{FB01}\u{20AC}\u{e9}\u{2D8}"
        );
    }
}
