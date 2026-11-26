//! Stream filters (ISO 32000-2 section 7.4): a stream's stored bytes to its data.
//!
//! Every lossless filter is decoded here (`ASCIIHexDecode`, `ASCII85Decode`, `LZWDecode`, `FlateDecode`, <!-- spelling-ignore-line: the PDF name -->
//! `RunLengthDecode`, with the PNG and TIFF predictors). An image codec (`DCTDecode`, `JPXDecode`, `JBIG2Decode`,
//! `CCITTFaxDecode`) is not decoded: the data stops there and is handed over with the codec's name and parameters,
//! for whoever turns it into pixels.

use crate::error::{Error, Result};
use crate::inflate::inflate;
use crate::object::{Dict, Object};

/// An image codec a stream's data is still encoded in.
#[derive(Debug, Clone, PartialEq)]
pub struct Codec {
    /// The filter's full name (`DCTDecode`, `JPXDecode`, `JBIG2Decode`, `CCITTFaxDecode`).
    pub name: Vec<u8>,
    /// Its `/DecodeParms`, when given. <!-- spelling-ignore-line: the PDF name -->
    pub params: Option<Dict>,
}

/// A stream's data after every lossless filter, and the image codec it is still in, if any.
#[derive(Debug, Clone, PartialEq)]
pub struct Decoded {
    /// The data.
    pub data: Vec<u8>,
    /// The image codec `data` is still encoded in.
    pub codec: Option<Codec>,
}

impl Decoded {
    /// The data, refusing one still in an image codec.
    ///
    /// # Errors
    ///
    /// Refuses input that is not what a PDF must be here; the [`Error`] says what and where.
    pub fn into_bytes(self) -> Result<Vec<u8>> {
        match self.codec {
            None => Ok(self.data),
            Some(c) => Err(Error::new(format!(
                "the data is {} image data, which is decoded to pixels, not to bytes",
                String::from_utf8_lossy(&c.name)
            ))),
        }
    }
}

/// Decode `raw` through the filters `dict` names; `resolve` follows a reference in `/Filter` or `/DecodeParms`. <!-- spelling-ignore-line: the PDF name -->
///
/// # Errors
///
/// Refuses input that is not what a PDF must be here; the [`Error`] says what and where.
pub fn decode(
    dict: &Dict,
    raw: &[u8],
    resolve: &dyn Fn(&Object) -> Result<Object>,
) -> Result<Decoded> {
    let filters = list(dict.get(b"Filter").or_else(|| dict.get(b"F")), resolve)?;
    let params = list(
        dict.get(b"DecodeParms").or_else(|| dict.get(b"DP")), // spelling-ignore-line: the PDF name
        resolve,
    )?;
    let mut data = raw.to_vec();
    for (i, filter) in filters.iter().enumerate() {
        let name = filter
            .as_name()
            .ok_or_else(|| Error::new(format!("a filter is a name, not a {}", filter.kind())))?;
        let param = match params.get(i).map(resolve).transpose()? {
            Some(Object::Dict(d)) => Some(d),
            _ => None,
        };
        data = match full_name(name) {
            b"ASCIIHexDecode" => ascii_hex(&data)?,
            b"ASCII85Decode" => ascii85(&data)?,
            b"LZWDecode" => predict(&lzw(&data, early_change(param.as_ref()))?, param.as_ref())?,
            b"FlateDecode" => predict(&inflate(&data)?, param.as_ref())?, // spelling-ignore-line: the PDF name
            b"RunLengthDecode" => run_length(&data)?,
            codec @ (b"DCTDecode" | b"JPXDecode" | b"JBIG2Decode" | b"CCITTFaxDecode") => {
                return Ok(Decoded {
                    data,
                    codec: Some(Codec {
                        name: codec.to_vec(),
                        params: param,
                    }),
                });
            }
            b"Crypt" => data, // the identity crypt filter; a named one is the security handler's, applied before
            other => {
                return Err(Error::new(format!(
                    "unknown filter /{}",
                    String::from_utf8_lossy(other)
                )));
            }
        };
    }
    Ok(Decoded { data, codec: None })
}

/// `/Filter` or `/DecodeParms` as a list: one value is a list of one. <!-- spelling-ignore-line: the PDF name -->
fn list(
    value: Option<&Object>,
    resolve: &dyn Fn(&Object) -> Result<Object>,
) -> Result<Vec<Object>> {
    Ok(match value.map(resolve).transpose()? {
        None | Some(Object::Null) => Vec::new(),
        Some(Object::Array(items)) => items,
        Some(one) => vec![one],
    })
}

/// The full name of an abbreviated filter name (inline images use the short forms).
fn full_name(name: &[u8]) -> &[u8] {
    match name {
        b"AHx" => b"ASCIIHexDecode",
        b"A85" => b"ASCII85Decode",
        b"LZW" => b"LZWDecode",
        b"Fl" => b"FlateDecode", // spelling-ignore-line: the PDF name
        b"RL" => b"RunLengthDecode",
        b"CCF" => b"CCITTFaxDecode",
        b"DCT" => b"DCTDecode",
        other => other,
    }
}

fn int(param: Option<&Dict>, key: &[u8], default: i64) -> i64 {
    param
        .and_then(|d| d.get(key))
        .and_then(Object::as_integer)
        .unwrap_or(default)
}

/// `param[key]` as a count of at least 1 (a missing, zero or negative value is `default`).
fn positive(param: Option<&Dict>, key: &[u8], default: usize) -> usize {
    usize::try_from(int(param, key, 0))
        .ok()
        .filter(|&v| v > 0)
        .unwrap_or(default)
}

fn early_change(param: Option<&Dict>) -> bool {
    int(param, b"EarlyChange", 1) != 0
}

fn ascii_hex(data: &[u8]) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(data.len() / 2);
    let mut high: Option<u8> = None;
    for (i, &b) in data.iter().enumerate() {
        let v = match b {
            b'>' => break,
            b'0'..=b'9' => b - b'0',
            b'a'..=b'f' => b - b'a' + 10,
            b'A'..=b'F' => b - b'A' + 10,
            _ if crate::parser::is_whitespace(b) => continue,
            _ => return Err(Error::at(i, "ASCIIHexDecode: not a hex digit")),
        };
        match high.take() {
            Some(h) => out.push(h << 4 | v),
            None => high = Some(v),
        }
    }
    if let Some(h) = high {
        out.push(h << 4);
    }
    Ok(out)
}

fn ascii85(data: &[u8]) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(data.len() * 4 / 5);
    let mut group = [0u8; 5];
    let mut n = 0;
    let mut i = 0;
    while i < data.len() {
        let b = data[i];
        i += 1;
        match b {
            b'~' => break,
            b'z' if n == 0 => out.extend_from_slice(&[0; 4]),
            b'!'..=b'u' => {
                group[n] = b - b'!';
                n += 1;
                if n == 5 {
                    out.extend_from_slice(&base85(group, i)?.to_be_bytes());
                    n = 0;
                }
            }
            _ if crate::parser::is_whitespace(b) => {}
            _ => return Err(Error::at(i - 1, "ASCII85Decode: not a base-85 digit")),
        }
    }
    if n == 1 {
        return Err(Error::at(i, "ASCII85Decode: a final group of one digit"));
    }
    if n > 1 {
        group[n..].fill(84);
        out.extend_from_slice(&base85(group, i)?.to_be_bytes()[..n - 1]);
    }
    Ok(out)
}

fn base85(group: [u8; 5], at: usize) -> Result<u32> {
    let v = group.iter().fold(0u64, |acc, &d| acc * 85 + u64::from(d));
    u32::try_from(v).map_err(|_| Error::at(at, "ASCII85Decode: a group exceeds 2^32"))
}

fn run_length(data: &[u8]) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(data.len() * 2);
    let mut i = 0;
    while let Some(&len) = data.get(i) {
        i += 1;
        match len {
            128 => break,
            0..=127 => {
                let run = data
                    .get(i..i + usize::from(len) + 1)
                    .ok_or_else(|| Error::at(i, "RunLengthDecode: run cut short"))?;
                out.extend_from_slice(run);
                i += usize::from(len) + 1;
            }
            _ => {
                let &b = data
                    .get(i)
                    .ok_or_else(|| Error::at(i, "RunLengthDecode: repeat cut short"))?;
                out.extend(std::iter::repeat_n(b, 257 - usize::from(len)));
                i += 1;
            }
        }
    }
    Ok(out)
}

fn lzw(data: &[u8], early: bool) -> Result<Vec<u8>> {
    const CLEAR: usize = 256;
    const END: usize = 257;
    let mut out = Vec::with_capacity(data.len() * 3);
    let mut table: Vec<(Option<usize>, u8, usize)> = Vec::new(); // (prefix code, last byte, length) per code
    let reset = |table: &mut Vec<(Option<usize>, u8, usize)>| {
        table.clear();
        table.extend((0..=255u8).map(|b| (None, b, 1)));
        table.push((None, 0, 0));
        table.push((None, 0, 0));
    };
    reset(&mut table);
    let (mut buf, mut count, mut pos) = (0u32, 0u32, 0usize);
    let mut width = 9u32;
    let mut prev: Option<usize> = None;
    loop {
        while count < width {
            let Some(&b) = data.get(pos) else {
                return Ok(out);
            }; // data may end without an EOD code
            buf = buf << 8 | u32::from(b);
            pos += 1;
            count += 8;
        }
        let code = ((buf >> (count - width)) & ((1 << width) - 1)) as usize;
        count -= width;
        match code {
            CLEAR => {
                reset(&mut table);
                width = 9;
                prev = None;
                continue;
            }
            END => return Ok(out),
            _ => {}
        }
        let start = out.len();
        match code.cmp(&table.len()) {
            std::cmp::Ordering::Less => {
                emit(&table, code, &mut out);
                if let Some(p) = prev {
                    table.push((Some(p), out[start], table[p].2 + 1));
                }
            }
            std::cmp::Ordering::Equal => {
                let p =
                    prev.ok_or_else(|| Error::at(pos, "LZWDecode: a new code with no previous"))?;
                emit(&table, p, &mut out);
                let first = out[start];
                out.push(first);
                table.push((Some(p), first, table[p].2 + 1));
            }
            std::cmp::Ordering::Greater => {
                return Err(Error::at(pos, "LZWDecode: a code past the table"));
            }
        }
        prev = Some(code);
        let next = table.len() + usize::from(early);
        width = match next {
            0..=511 => 9,
            512..=1023 => 10,
            1024..=2047 => 11,
            _ => 12,
        };
    }
}

fn emit(table: &[(Option<usize>, u8, usize)], code: usize, out: &mut Vec<u8>) {
    let len = table[code].2;
    let start = out.len();
    out.resize(start + len, 0);
    let mut c = Some(code);
    let mut i = start + len;
    while let Some(k) = c {
        i -= 1;
        out[i] = table[k].1;
        c = table[k].0;
    }
}

/// Undo `/Predictor` (2: TIFF; 10-15: PNG, each row tagged with its own filter).
fn predict(data: &[u8], param: Option<&Dict>) -> Result<Vec<u8>> {
    let predictor = int(param, b"Predictor", 1);
    if predictor == 1 {
        return Ok(data.to_vec());
    }
    let colors = positive(param, b"Colors", 1);
    let bpc = positive(param, b"BitsPerComponent", 8);
    let columns = positive(param, b"Columns", 1);
    let bpp = (colors * bpc).div_ceil(8);
    let row = (colors * bpc * columns).div_ceil(8);
    match predictor {
        2 => tiff(data, row, colors, bpc),
        10..=15 => png(data, row, bpp),
        other => Err(Error::new(format!("unknown predictor {other}"))),
    }
}

fn tiff(data: &[u8], row: usize, colors: usize, bpc: usize) -> Result<Vec<u8>> {
    if bpc != 8 {
        return Err(Error::new(format!(
            "TIFF predictor with {bpc} bits per component is not read"
        )));
    }
    let mut out = data.to_vec();
    for line in out.chunks_mut(row) {
        for i in colors..line.len() {
            line[i] = line[i].wrapping_add(line[i - colors]);
        }
    }
    Ok(out)
}

fn png(data: &[u8], row: usize, bpp: usize) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(data.len());
    let mut prior = vec![0u8; row];
    for (n, chunk) in data.chunks(row + 1).enumerate() {
        let (&kind, bytes) = chunk
            .split_first()
            .ok_or_else(|| Error::new("PNG predictor: empty row"))?;
        let mut line = bytes.to_vec();
        line.resize(row, 0); // a last row cut short is completed with zeros
        for i in 0..row {
            let left = if i >= bpp { line[i - bpp] } else { 0 };
            let up = prior[i];
            let up_left = if i >= bpp { prior[i - bpp] } else { 0 };
            line[i] = line[i].wrapping_add(match kind {
                0 => 0,
                1 => left,
                2 => up,
                3 => u8::midpoint(left, up),
                4 => nearest(left, up, up_left),
                _ => {
                    return Err(Error::new(format!(
                        "PNG predictor: unknown row filter {kind} on row {n}"
                    )));
                }
            });
        }
        out.extend_from_slice(&line[..bytes.len().min(row)]);
        prior = line;
    }
    Ok(out)
}

/// PNG filter type 4: of left, up and up-left, the one nearest to left + up - up-left.
fn nearest(a: u8, b: u8, c: u8) -> u8 {
    let p = i16::from(a) + i16::from(b) - i16::from(c);
    let (pa, pb, pc) = (
        (p - i16::from(a)).abs(),
        (p - i16::from(b)).abs(),
        (p - i16::from(c)).abs(),
    );
    if pa <= pb && pa <= pc {
        a
    } else if pb <= pc {
        b
    } else {
        c
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::Parser;

    fn dict(text: &[u8]) -> Dict {
        match Parser::new(text, 0).object().unwrap() {
            Object::Dict(d) => d,
            other => panic!("{other:?}"),
        }
    }

    /// A resolver for direct objects only: every value is already itself.
    const DIRECT: &dyn Fn(&Object) -> Result<Object> = &|o| Ok(o.clone());

    fn run(filters: &[u8], raw: &[u8]) -> Result<Vec<u8>> {
        decode(&dict(filters), raw, DIRECT)?.into_bytes()
    }

    #[test]
    fn ascii_hex_and_ascii85_decode() {
        assert_eq!(
            run(b"<< /Filter /AHx >>", b"48 65 6C6c 6F>").unwrap(),
            b"Hello"
        );
        // "Man is" -> 9jqo^Bla (a final partial group), and the `z` group of zeros.
        assert_eq!(
            run(b"<< /Filter /ASCII85Decode >>", b"9jqo^Bla~>").unwrap(),
            b"Man is"
        );
        assert_eq!(run(b"<< /Filter /A85 >>", b"z~>").unwrap(), [0, 0, 0, 0]);
    }

    #[test]
    fn run_length_decodes() {
        assert_eq!(
            run(
                b"<< /Filter /RunLengthDecode >>",
                &[2, b'a', b'b', b'c', 254, b'x', 128]
            )
            .unwrap(),
            b"abcxxx"
        );
    }

    #[test]
    fn lzw_decodes_the_spec_example() {
        // ISO 32000-2 7.4.4.2: 45 45 45 45 45 65 45 45 45 66 encodes to 80 0B 60 50 22 0C 0C 85 01.
        let data = [0x80, 0x0B, 0x60, 0x50, 0x22, 0x0C, 0x0C, 0x85, 0x01];
        assert_eq!(
            run(b"<< /Filter /LZWDecode >>", &data).unwrap(),
            [45, 45, 45, 45, 45, 65, 45, 45, 45, 66]
        );
    }

    #[test]
    fn deflate_with_a_png_predictor_undoes_each_row() {
        // Two rows of 3 bytes: row 1 filter Sub (1), row 2 filter Up (2); stored deflate.
        let rows = [1u8, 10, 5, 5, 2, 1, 1, 1];
        let len = u8::try_from(rows.len()).unwrap();
        let mut zlib = vec![0x78, 0x01, 0x01, len, 0, !len, 0xff];
        zlib.extend_from_slice(&rows);
        let out = run(
            b"<< /Filter /FlateDecode /DecodeParms << /Predictor 12 /Columns 3 >> >>", // spelling-ignore-line: the PDF name
            &zlib,
        )
        .unwrap();
        assert_eq!(out, [10, 15, 20, 11, 16, 21]);
    }

    #[test]
    fn a_filter_chain_applies_in_order() {
        assert_eq!(
            run(b"<< /Filter [/AHx /RL] >>", b"02616263FE7880>").unwrap(),
            b"abcxxx"
        );
    }

    #[test]
    fn an_image_codec_is_handed_over_undecoded() {
        let got = decode(&dict(b"<< /Filter [/AHx /DCTDecode] >>"), b"FFD8>", DIRECT).unwrap();
        assert_eq!(got.data, [0xFF, 0xD8]);
        assert_eq!(
            got.codec.as_ref().map(|c| c.name.as_slice()),
            Some(&b"DCTDecode"[..])
        );
        assert!(got.into_bytes().is_err());
    }

    #[test]
    fn an_unknown_filter_is_refused() {
        assert!(run(b"<< /Filter /Nope >>", b"x").is_err());
    }
}
