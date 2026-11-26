//! Bytes to objects (ISO 32000-2 section 7.2-7.3): the lexer and the object parser over one byte slice.

use crate::error::{Error, Result};
use crate::object::{Dict, ObjRef, Object, Stream};

/// Whether `b` is PDF white-space.
#[must_use]
pub fn is_whitespace(b: u8) -> bool {
    matches!(b, b'\0' | b'\t' | b'\n' | b'\x0c' | b'\r' | b' ')
}

/// Whether `b` is a PDF delimiter.
#[must_use]
pub fn is_delimiter(b: u8) -> bool {
    matches!(
        b,
        b'(' | b')' | b'<' | b'>' | b'[' | b']' | b'{' | b'}' | b'/' | b'%'
    )
}

fn is_regular(b: u8) -> bool {
    !is_whitespace(b) && !is_delimiter(b)
}

/// How deep arrays and dictionaries may nest before the input is refused as hostile.
const MAX_DEPTH: usize = 256;

/// A cursor over `data`, reading tokens and objects from `pos` on.
pub struct Parser<'a> {
    data: &'a [u8],
    /// The next byte to read.
    pub pos: usize,
}

impl<'a> Parser<'a> {
    /// A parser at `pos` in `data`.
    #[must_use]
    pub fn new(data: &'a [u8], pos: usize) -> Self {
        Self { data, pos }
    }

    /// The whole data this parser reads.
    #[must_use]
    pub fn data(&self) -> &'a [u8] {
        self.data
    }

    /// Skip white-space and comments.
    pub fn skip_whitespace(&mut self) {
        while let Some(&b) = self.data.get(self.pos) {
            if is_whitespace(b) {
                self.pos += 1;
            } else if b == b'%' {
                while let Some(&c) = self.data.get(self.pos) {
                    if c == b'\r' || c == b'\n' {
                        break;
                    }
                    self.pos += 1;
                }
            } else {
                break;
            }
        }
    }

    /// The next run of regular characters (a keyword or a number), white-space skipped first.
    pub fn token(&mut self) -> &'a [u8] {
        self.skip_whitespace();
        let start = self.pos;
        while self.data.get(self.pos).is_some_and(|&b| is_regular(b)) {
            self.pos += 1;
        }
        &self.data[start..self.pos]
    }

    /// Consume the keyword `word`, or refuse.
    ///
    /// # Errors
    ///
    /// Refuses input that is not what a PDF must be here; the [`Error`] says what and where.
    pub fn expect(&mut self, word: &[u8]) -> Result<()> {
        let at = self.pos;
        let got = self.token();
        if got == word {
            Ok(())
        } else {
            Err(Error::at(
                at,
                format!("expected `{}`, found `{}`", show(word), show(got)),
            ))
        }
    }

    /// The next unsigned integer token, or refuse.
    ///
    /// # Errors
    ///
    /// Refuses input that is not what a PDF must be here; the [`Error`] says what and where.
    pub fn unsigned(&mut self) -> Result<u64> {
        let at = self.pos;
        let got = self.token();
        parse_unsigned(got).ok_or_else(|| {
            Error::at(
                at,
                format!("expected an unsigned integer, found `{}`", show(got)),
            )
        })
    }

    /// The next object (a reference `n g R` included).
    ///
    /// # Errors
    ///
    /// Refuses input that is not what a PDF must be here; the [`Error`] says what and where.
    pub fn object(&mut self) -> Result<Object> {
        self.object_at_depth(0)
    }

    fn object_at_depth(&mut self, depth: usize) -> Result<Object> {
        if depth > MAX_DEPTH {
            return Err(Error::at(
                self.pos,
                "arrays and dictionaries nested too deep",
            ));
        }
        self.skip_whitespace();
        let at = self.pos;
        let Some(&b) = self.data.get(at) else {
            return Err(Error::at(at, "end of data where an object was expected"));
        };
        match b {
            b'/' => {
                self.pos += 1;
                Ok(Object::Name(self.name_body()))
            }
            b'(' => {
                self.pos += 1;
                self.literal_string().map(Object::String)
            }
            b'<' if self.data.get(at + 1) == Some(&b'<') => {
                self.pos += 2;
                self.dict_body(depth).map(Object::Dict)
            }
            b'<' => {
                self.pos += 1;
                self.hex_string().map(Object::String)
            }
            b'[' => {
                self.pos += 1;
                let mut items = Vec::new();
                loop {
                    self.skip_whitespace();
                    match self.data.get(self.pos) {
                        Some(b']') => {
                            self.pos += 1;
                            return Ok(Object::Array(items));
                        }
                        None => return Err(Error::at(at, "array not closed")),
                        _ => items.push(self.object_at_depth(depth + 1)?),
                    }
                }
            }
            b'+' | b'-' | b'.' | b'0'..=b'9' => self.number_or_reference(),
            _ => {
                let word = self.token();
                match word {
                    b"true" => Ok(Object::Bool(true)),
                    b"false" => Ok(Object::Bool(false)),
                    b"null" => Ok(Object::Null),
                    b"" => Err(Error::at(at, format!("unexpected `{}`", show(&[b])))),
                    _ => Err(Error::at(
                        at,
                        format!("unexpected keyword `{}`", show(word)),
                    )),
                }
            }
        }
    }

    fn number_or_reference(&mut self) -> Result<Object> {
        let at = self.pos;
        let word = self.token();
        let number = parse_number(word)
            .ok_or_else(|| Error::at(at, format!("not a number: `{}`", show(word))))?;
        if let Object::Integer(num) = number
            && let (Ok(num), true) = (u32::try_from(num), word[0].is_ascii_digit())
        {
            let after = self.pos;
            if let Some(generation) =
                parse_unsigned(self.token()).and_then(|g| u16::try_from(g).ok())
                && self.token() == b"R"
            {
                return Ok(Object::Reference(ObjRef { num, generation }));
            }
            self.pos = after;
        }
        Ok(number)
    }

    fn name_body(&mut self) -> Vec<u8> {
        let mut out = Vec::new();
        while let Some(&b) = self.data.get(self.pos) {
            if !is_regular(b) {
                break;
            }
            if b == b'#'
                && let Some(v) = self.data.get(self.pos + 1..self.pos + 3).and_then(hex_pair)
            {
                out.push(v);
                self.pos += 3;
                continue;
            }
            out.push(b);
            self.pos += 1;
        }
        out
    }

    fn literal_string(&mut self) -> Result<Vec<u8>> {
        let start = self.pos - 1;
        let mut out = Vec::new();
        let mut depth = 1usize;
        loop {
            let Some(&b) = self.data.get(self.pos) else {
                return Err(Error::at(start, "string not closed"));
            };
            self.pos += 1;
            match b {
                b'(' => {
                    depth += 1;
                    out.push(b);
                }
                b')' => {
                    depth -= 1;
                    if depth == 0 {
                        return Ok(out);
                    }
                    out.push(b);
                }
                b'\r' => {
                    // An end-of-line in a string is read as a single LF, whatever its form.
                    if self.data.get(self.pos) == Some(&b'\n') {
                        self.pos += 1;
                    }
                    out.push(b'\n');
                }
                b'\\' => self.escape(&mut out),
                _ => out.push(b),
            }
        }
    }

    fn escape(&mut self, out: &mut Vec<u8>) {
        let Some(&e) = self.data.get(self.pos) else {
            return;
        };
        self.pos += 1;
        match e {
            b'n' => out.push(b'\n'),
            b'r' => out.push(b'\r'),
            b't' => out.push(b'\t'),
            b'b' => out.push(0x08),
            b'f' => out.push(0x0c),
            b'0'..=b'7' => {
                let mut v = u32::from(e - b'0');
                for _ in 0..2 {
                    match self.data.get(self.pos) {
                        Some(&d @ b'0'..=b'7') => {
                            v = v * 8 + u32::from(d - b'0');
                            self.pos += 1;
                        }
                        _ => break,
                    }
                }
                out.push((v & 0xff) as u8);
            }
            b'\r' => {
                // A backslash before an end-of-line continues the string on the next line.
                if self.data.get(self.pos) == Some(&b'\n') {
                    self.pos += 1;
                }
            }
            b'\n' => {}
            // `\(`, `\)`, `\\` are the character; an unknown escape is the character too (the backslash is ignored).
            _ => out.push(e),
        }
    }

    fn hex_string(&mut self) -> Result<Vec<u8>> {
        let start = self.pos - 1;
        let mut out = Vec::new();
        let mut high: Option<u8> = None;
        loop {
            let Some(&b) = self.data.get(self.pos) else {
                return Err(Error::at(start, "hex string not closed"));
            };
            self.pos += 1;
            if b == b'>' {
                if let Some(h) = high {
                    out.push(h << 4); // an odd digit count: the last digit is followed by 0
                }
                return Ok(out);
            }
            if is_whitespace(b) {
                continue;
            }
            let v = hex_digit(b).ok_or_else(|| {
                Error::at(self.pos - 1, format!("not a hex digit: `{}`", show(&[b])))
            })?;
            match high.take() {
                Some(h) => out.push((h << 4) | v),
                None => high = Some(v),
            }
        }
    }

    fn dict_body(&mut self, depth: usize) -> Result<Dict> {
        let start = self.pos - 2;
        let mut dict = Dict::new();
        loop {
            self.skip_whitespace();
            match self.data.get(self.pos) {
                Some(b'>') if self.data.get(self.pos + 1) == Some(&b'>') => {
                    self.pos += 2;
                    return Ok(dict);
                }
                Some(b'/') => {
                    self.pos += 1;
                    let key = self.name_body();
                    let value = self.object_at_depth(depth + 1)?;
                    // A null value is the same as the key being absent (7.3.7).
                    if value != Object::Null {
                        dict.insert(key, value);
                    }
                }
                None => return Err(Error::at(start, "dictionary not closed")),
                Some(_) => return Err(Error::at(self.pos, "a dictionary key must be a name")),
            }
        }
    }

    /// The indirect object `num gen obj ... endobj` here: its reference and its value.
    ///
    /// A stream's length is `length` of its `/Length` value (a direct integer, or a reference the caller resolves);
    /// when it is unknown or wrong, the data runs to the next `endstream`.
    ///
    /// # Errors
    ///
    /// Refuses input that is not what a PDF must be here; the [`Error`] says what and where.
    pub fn indirect_object(
        &mut self,
        length: impl Fn(&Object) -> Option<usize>,
    ) -> Result<(ObjRef, Object)> {
        let at = self.pos;
        let num = self.unsigned()?;
        let generation = self.unsigned()?;
        self.expect(b"obj")?;
        let num = u32::try_from(num).map_err(|_| Error::at(at, "object number out of range"))?;
        let generation = u16::try_from(generation)
            .map_err(|_| Error::at(at, "generation number out of range"))?;
        let id = ObjRef { num, generation };
        let value = self.object()?;
        let value = match value {
            Object::Dict(dict) => {
                let before = self.pos;
                if self.token() == b"stream" {
                    let declared = dict.get(b"Length").and_then(&length);
                    Object::Stream(Stream {
                        raw: self.stream_data(declared)?,
                        dict,
                    })
                } else {
                    self.pos = before;
                    Object::Dict(dict)
                }
            }
            other => other,
        };
        Ok((id, value))
    }

    fn stream_data(&mut self, declared: Option<usize>) -> Result<Vec<u8>> {
        // `stream` is followed by CRLF or LF (a lone CR is tolerated).
        match self.data.get(self.pos) {
            Some(b'\r') if self.data.get(self.pos + 1) == Some(&b'\n') => self.pos += 2,
            Some(b'\r' | b'\n') => self.pos += 1,
            _ => {}
        }
        let start = self.pos;
        if let Some(end) = declared
            .and_then(|len| start.checked_add(len))
            .filter(|&e| e <= self.data.len())
        {
            let mut after = Parser::new(self.data, end);
            if after.token() == b"endstream" {
                self.pos = after.pos;
                return Ok(self.data[start..end].to_vec());
            }
        }
        let found = find(&self.data[start..], b"endstream")
            .ok_or_else(|| Error::at(start, "stream without `endstream`"))?;
        let mut end = start + found;
        self.pos = end + b"endstream".len();
        // The end-of-line before `endstream` is not data.
        if end > start && self.data[end - 1] == b'\n' {
            end -= 1;
        }
        if end > start && self.data[end - 1] == b'\r' {
            end -= 1;
        }
        Ok(self.data[start..end].to_vec())
    }
}

/// The first offset of `needle` in `hay`.
#[must_use]
pub fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

/// The last offset of `needle` in `hay`.
#[must_use]
pub fn rfind(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).rposition(|w| w == needle)
}

fn parse_unsigned(word: &[u8]) -> Option<u64> {
    if word.is_empty() || !word.iter().all(u8::is_ascii_digit) {
        return None;
    }
    std::str::from_utf8(word).ok()?.parse().ok()
}

fn parse_number(word: &[u8]) -> Option<Object> {
    let text = std::str::from_utf8(word).ok()?;
    let digits = text.trim_start_matches(['+', '-']);
    if digits.is_empty()
        || !digits.bytes().all(|b| b.is_ascii_digit() || b == b'.')
        || digits.matches('.').count() > 1
    {
        return None;
    }
    let negative = text.starts_with('-');
    if !digits.contains('.') {
        // An integer past i64 is read as a real, as a reader must not fail on it.
        return Some(match digits.parse::<i64>() {
            Ok(v) => Object::Integer(if negative { -v } else { v }),
            Err(_) => Object::Real(digits.parse::<f64>().ok()? * if negative { -1.0 } else { 1.0 }),
        });
    }
    let v: f64 = if digits == "." {
        0.0
    } else {
        digits.parse().ok()?
    };
    Some(Object::Real(if negative { -v } else { v }))
}

fn hex_digit(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

fn hex_pair(pair: &[u8]) -> Option<u8> {
    Some((hex_digit(pair[0])? << 4) | hex_digit(pair[1])?)
}

/// Bytes shown for an error message.
#[must_use]
pub fn show(bytes: &[u8]) -> String {
    String::from_utf8_lossy(&bytes[..bytes.len().min(40)]).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(text: &[u8]) -> Object {
        Parser::new(text, 0).object().unwrap()
    }

    fn name(n: &[u8]) -> Object {
        Object::Name(n.to_vec())
    }

    #[test]
    fn scalars_read_as_their_values() {
        assert_eq!(parse(b"true"), Object::Bool(true));
        assert_eq!(parse(b"null"), Object::Null);
        assert_eq!(parse(b"-42"), Object::Integer(-42));
        assert_eq!(parse(b"+17"), Object::Integer(17));
        assert_eq!(parse(b"4."), Object::Real(4.0));
        assert_eq!(parse(b"-.5"), Object::Real(-0.5));
        assert_eq!(parse(b"99999999999999999999"), Object::Real(1e20));
    }

    #[test]
    fn a_name_resolves_its_hex_escapes() {
        assert_eq!(parse(b"/A#20B"), name(b"A B"));
        assert_eq!(parse(b"/Type/Page"), name(b"Type"));
    }

    #[test]
    fn a_literal_string_reads_escapes_balanced_parens_and_line_ends() {
        assert_eq!(parse(b"(a(b)c)"), Object::String(b"a(b)c".to_vec()));
        assert_eq!(
            parse(br"(\n\(\)\\\101\0539)"),
            Object::String(b"\n()\\A+9".to_vec())
        );
        assert_eq!(parse(b"(a\\\r\nb)"), Object::String(b"ab".to_vec()));
        assert_eq!(parse(b"(a\r\nb\rc)"), Object::String(b"a\nb\nc".to_vec()));
    }

    #[test]
    fn a_hex_string_pads_an_odd_digit_count() {
        assert_eq!(parse(b"<48 65 6c6C 6>"), Object::String(b"Hell`".to_vec()));
    }

    #[test]
    fn a_reference_is_told_from_two_integers() {
        assert_eq!(
            parse(b"12 0 R"),
            Object::Reference(ObjRef {
                num: 12,
                generation: 0
            })
        );
        let Object::Array(items) = parse(b"[1 2 3 0 R 4]") else {
            panic!()
        };
        assert_eq!(
            items,
            vec![
                Object::Integer(1),
                Object::Integer(2),
                Object::Reference(ObjRef {
                    num: 3,
                    generation: 0
                }),
                Object::Integer(4)
            ]
        );
    }

    #[test]
    fn a_dictionary_keeps_order_drops_nulls_and_skips_comments() {
        let Object::Dict(d) = parse(b"<< /B 1 % note\n /A [/X] /N null >>") else {
            panic!()
        };
        assert_eq!(
            d.iter().map(|(k, _)| k.to_vec()).collect::<Vec<_>>(),
            vec![b"B".to_vec(), b"A".to_vec()]
        );
        assert_eq!(d.get(b"A"), Some(&Object::Array(vec![name(b"X")])));
    }

    #[test]
    fn malformed_input_is_refused_with_its_offset() {
        let err = Parser::new(b"[1 2", 0).object().unwrap_err();
        assert_eq!(err.offset, Some(0));
        assert!(Parser::new(b"<< 1 2 >>", 0).object().is_err());
        assert!(Parser::new(b"<4G>", 0).object().is_err());
        let deep = vec![b'['; MAX_DEPTH + 2];
        assert!(Parser::new(&deep, 0).object().is_err());
    }

    #[test]
    fn a_stream_takes_its_declared_length_or_runs_to_endstream() {
        let text = b"7 0 obj << /Length 5 >> stream\r\nab\ncd\nendstream endobj";
        let (id, obj) = Parser::new(text, 0)
            .indirect_object(|o| o.as_integer().and_then(|v| usize::try_from(v).ok()))
            .unwrap();
        assert_eq!(
            id,
            ObjRef {
                num: 7,
                generation: 0
            }
        );
        let Object::Stream(s) = obj else { panic!() };
        assert_eq!(s.raw, b"ab\ncd");
        // A wrong length: the data is what lies before `endstream`, less its end-of-line.
        let (_, obj) = Parser::new(text, 0).indirect_object(|_| Some(2)).unwrap();
        let Object::Stream(s) = obj else { panic!() };
        assert_eq!(s.raw, b"ab\ncd");
    }
}
