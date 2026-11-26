//! Where each object is (ISO 32000-2 section 7.5): the cross-reference sections and the trailer.
//!
//! [`read`] follows `startxref` back through every `/Prev` section -- tables, cross-reference streams and hybrid
//! files alike -- the newest entry for an object winning. [`scan`] rebuilds the table from the objects found in
//! the file, for a file whose sections are missing or wrong.

use std::collections::{BTreeMap, HashSet};

use crate::error::{Error, Result};
use crate::filter;
use crate::object::{Dict, Object};
use crate::parser::{Parser, find, rfind};

/// Where one object is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Entry {
    /// Deleted, or never there.
    Free,
    /// At a byte offset of the file.
    InFile {
        /// The offset of `num gen obj`.
        offset: usize,
        /// The generation number.
        generation: u16,
    },
    /// Inside an object stream.
    InStream {
        /// The object stream's number.
        stream: u32,
        /// The object's index within it.
        index: u32,
    },
}

/// The cross-reference table and the trailer it ends with.
#[derive(Debug, Clone, Default)]
pub struct Xref {
    /// Every object number's entry.
    pub entries: BTreeMap<u32, Entry>,
    /// The trailer dictionary (for a cross-reference stream, its dictionary).
    pub trailer: Dict,
}

impl Xref {
    fn add(&mut self, num: u32, entry: Entry) {
        self.entries.entry(num).or_insert(entry);
    }

    fn merge_trailer(&mut self, older: &Dict) {
        for (k, v) in older.iter() {
            self.trailer.insert(k.to_vec(), v.clone());
        }
    }
}

/// How many sections a `/Prev` chain may hold before it is refused as a loop or a hostile file.
const MAX_SECTIONS: usize = 4096;

/// The table read from `startxref` back through every `/Prev`.
///
/// # Errors
///
/// Refuses input that is not what a PDF must be here; the [`Error`] says what and where.
pub fn read(data: &[u8]) -> Result<Xref> {
    let tail = data.len().saturating_sub(2048);
    let at = rfind(&data[tail..], b"startxref")
        .ok_or_else(|| Error::new("no `startxref` near the end of the file"))?;
    let mut parser = Parser::new(data, tail + at + b"startxref".len());
    let mut next = Some(
        usize::try_from(parser.unsigned()?).map_err(|_| Error::new("startxref out of range"))?,
    );
    let mut xref = Xref::default();
    let mut seen = HashSet::new();
    while let Some(offset) = next.take() {
        if !seen.insert(offset) || seen.len() > MAX_SECTIONS {
            return Err(Error::at(offset, "the `/Prev` chain loops"));
        }
        let trailer = section(data, offset, &mut xref)?;
        next = trailer
            .get(b"Prev")
            .and_then(Object::as_integer)
            .map(offset_of)
            .transpose()?;
        xref.merge_trailer(&trailer);
    }
    Ok(xref)
}

fn offset_of(v: i64) -> Result<usize> {
    usize::try_from(v).map_err(|_| Error::new(format!("offset {v} out of range")))
}

/// Read the section at `offset` into `xref` (older entries never overwrite newer ones); its trailer.
fn section(data: &[u8], offset: usize, xref: &mut Xref) -> Result<Dict> {
    if offset >= data.len() {
        return Err(Error::at(
            offset,
            "a cross-reference offset past the end of the file",
        ));
    }
    let mut parser = Parser::new(data, offset);
    if parser.token() == b"xref" {
        return table(&mut parser, xref);
    }
    stream(data, offset, xref)
}

fn table(parser: &mut Parser<'_>, xref: &mut Xref) -> Result<Dict> {
    let mut found = Vec::new();
    loop {
        let at = parser.pos;
        let word = parser.token();
        if word == b"trailer" {
            break;
        }
        parser.pos = at;
        let first = parser.unsigned()?;
        let count = parser.unsigned()?;
        let mut rows = Vec::new();
        for _ in 0..count {
            let offset = parser.unsigned()?;
            let generation = parser.unsigned()?;
            let at = parser.pos;
            let kind = parser.token();
            rows.push((offset, generation, kind == b"n"));
            if kind != b"n" && kind != b"f" {
                return Err(Error::at(
                    at,
                    "a cross-reference entry is neither `n` nor `f`",
                ));
            }
        }
        // A common writer error: a section starting at 1 whose first entry is object 0's free head.
        let first = if first == 1 && rows.first() == Some(&(0, 65535, false)) {
            0
        } else {
            first
        };
        for (i, (offset, generation, used)) in rows.into_iter().enumerate() {
            let num = u32::try_from(first + i as u64)
                .map_err(|_| Error::new("object number out of range"))?;
            let entry = if used {
                Entry::InFile {
                    offset: usize::try_from(offset)
                        .map_err(|_| Error::new("offset out of range"))?,
                    generation: u16::try_from(generation).unwrap_or(u16::MAX),
                }
            } else {
                Entry::Free
            };
            found.push((num, entry));
        }
    }
    let trailer = match parser.object()? {
        Object::Dict(d) => d,
        other => {
            return Err(Error::at(
                parser.pos,
                format!("the trailer is a {}, not a dictionary", other.kind()),
            ));
        }
    };
    // A hybrid file's stream entries come first: the table marks those objects free for older readers.
    if let Some(stm) = trailer.get(b"XRefStm").and_then(Object::as_integer) {
        let _ = stream(parser_data(parser), offset_of(stm)?, xref);
    }
    for (num, entry) in found {
        xref.add(num, entry);
    }
    Ok(trailer)
}

fn parser_data<'a>(parser: &Parser<'a>) -> &'a [u8] {
    parser.data()
}

fn stream(data: &[u8], offset: usize, xref: &mut Xref) -> Result<Dict> {
    let (_, object) = Parser::new(data, offset)
        .indirect_object(|o| o.as_integer().and_then(|v| usize::try_from(v).ok()))?;
    let Object::Stream(s) = object else {
        return Err(Error::at(
            offset,
            "neither `xref` nor a cross-reference stream at this offset",
        ));
    };
    if s.dict.get(b"Type").and_then(Object::as_name) != Some(b"XRef") {
        return Err(Error::at(offset, "the stream here is not /Type /XRef"));
    }
    let bytes = filter::decode(&s.dict, &s.raw, &|o| Ok(o.clone()))?.into_bytes()?;
    let widths: Vec<usize> = s
        .dict
        .get(b"W")
        .and_then(Object::as_array)
        .ok_or_else(|| Error::at(offset, "a cross-reference stream without /W"))?
        .iter()
        .map(|w| {
            w.as_integer()
                .and_then(|v| usize::try_from(v).ok())
                .filter(|&v| v <= 8)
        })
        .collect::<Option<_>>()
        .filter(|w: &Vec<usize>| w.len() == 3)
        .ok_or_else(|| Error::at(offset, "/W is not three byte widths"))?;
    let size = s
        .dict
        .get(b"Size")
        .and_then(Object::as_integer)
        .unwrap_or(0);
    let index: Vec<i64> = match s.dict.get(b"Index").and_then(Object::as_array) {
        Some(a) => a.iter().filter_map(Object::as_integer).collect(),
        None => vec![0, size],
    };
    let row = widths.iter().sum::<usize>();
    let mut rows = bytes.chunks_exact(row.max(1));
    for pair in index.as_chunks::<2>().0 {
        for i in 0..pair[1] {
            let Some(r) = rows.next() else { break };
            let num =
                u32::try_from(pair[0] + i).map_err(|_| Error::new("object number out of range"))?;
            let (f0, rest) = r.split_at(widths[0]);
            let (f1, f2) = rest.split_at(widths[1]);
            // A zero-width type field means type 1.
            let kind = if widths[0] == 0 { 1 } else { be(f0) };
            let entry = match kind {
                0 => Entry::Free,
                1 => Entry::InFile {
                    offset: usize::try_from(be(f1))
                        .map_err(|_| Error::new("offset out of range"))?,
                    generation: u16::try_from(be(f2)).unwrap_or(u16::MAX),
                },
                2 => Entry::InStream {
                    stream: u32::try_from(be(f1))
                        .map_err(|_| Error::new("object stream number out of range"))?,
                    index: u32::try_from(be(f2)).unwrap_or(u32::MAX),
                },
                _ => continue, // a type this reader does not know is a null object (7.5.8.3)
            };
            xref.add(num, entry);
        }
    }
    Ok(s.dict)
}

fn be(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0u64, |acc, &b| acc << 8 | u64::from(b))
}

/// The table rebuilt from every `num gen obj` in the file (the last one of a number winning, as an incremental
/// update appends), and the trailer from every `trailer` dictionary (the last one winning) and every
/// cross-reference stream's dictionary.
pub fn scan(data: &[u8]) -> Xref {
    let mut found: BTreeMap<u32, Entry> = BTreeMap::new();
    let mut trailers = Vec::new();
    let mut i = 0;
    while let Some(at) = find(&data[i..], b"obj") {
        let end = i + at;
        i = end + 3;
        if data
            .get(i)
            .is_some_and(|&b| !crate::parser::is_whitespace(b) && !crate::parser::is_delimiter(b))
        {
            continue; // `objects`, `endobj`'s tail is handled below
        }
        if end >= 3 && &data[end - 3..end] == b"end" {
            continue;
        }
        if let Some((num, generation, start)) = header_before(data, end) {
            found.insert(
                num,
                Entry::InFile {
                    offset: start,
                    generation,
                },
            );
            let mut parser = Parser::new(data, start);
            if let Ok((_, Object::Stream(s))) =
                parser.indirect_object(|o| o.as_integer().and_then(|v| usize::try_from(v).ok()))
                && s.dict.get(b"Type").and_then(Object::as_name) == Some(b"XRef")
            {
                trailers.push(s.dict);
            }
        }
    }
    let mut j = 0;
    while let Some(at) = find(&data[j..], b"trailer") {
        let mut parser = Parser::new(data, j + at + b"trailer".len());
        j += at + b"trailer".len();
        if let Ok(Object::Dict(d)) = parser.object() {
            trailers.push(d);
        }
    }
    let mut xref = Xref {
        entries: found,
        trailer: Dict::new(),
    };
    for t in trailers.iter().rev() {
        xref.merge_trailer(t);
    }
    xref
}

/// `num gen` ending just before `obj` at `end`, and where `num` starts; None when the bytes are not that.
fn header_before(data: &[u8], end: usize) -> Option<(u32, u16, usize)> {
    let mut k = end;
    let digits_back = |k: &mut usize| -> Option<u64> {
        while *k > 0 && crate::parser::is_whitespace(data[*k - 1]) {
            *k -= 1;
        }
        let stop = *k;
        while *k > 0 && data[*k - 1].is_ascii_digit() {
            *k -= 1;
        }
        std::str::from_utf8(&data[*k..stop]).ok()?.parse().ok()
    };
    let generation = digits_back(&mut k)?;
    let num = digits_back(&mut k)?;
    if k > 0
        && !crate::parser::is_whitespace(data[k - 1])
        && !crate::parser::is_delimiter(data[k - 1])
    {
        return None;
    }
    Some((u32::try_from(num).ok()?, u16::try_from(generation).ok()?, k))
}
