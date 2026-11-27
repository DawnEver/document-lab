//! DEFLATE (RFC 1951) inside an optional zlib wrapper (RFC 1950): what `/FlateDecode` holds. <!-- spelling-ignore-line: the PDF name -->

use crate::error::{Error, Result};

const LENGTH_BASE: [u16; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131,
    163, 195, 227, 258,
];
const LENGTH_EXTRA: [u8; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];
const DIST_BASE: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
    2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
];
const DIST_EXTRA: [u8; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
    13,
];
/// The order code-length code lengths are stored in (3.2.7).
const CODE_LENGTH_ORDER: [usize; 19] = [
    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
];

struct Bits<'a> {
    data: &'a [u8],
    pos: usize,
    buf: u32,
    count: u32,
}

impl Bits<'_> {
    fn take(&mut self, n: u32) -> Result<u32> {
        while self.count < n {
            let byte = *self
                .data
                .get(self.pos)
                .ok_or_else(|| Error::malformed_at(self.pos, "deflate data ends early"))?;
            self.buf |= u32::from(byte) << self.count;
            self.pos += 1;
            self.count += 8;
        }
        let v = self.buf & ((1u32 << n) - 1);
        self.buf >>= n;
        self.count -= n;
        Ok(v)
    }

    fn align(&mut self) {
        self.buf = 0;
        self.count = 0;
    }
}

/// A canonical Huffman code: how many codes of each length, and the symbols in code order.
struct Huffman {
    counts: [u16; 16],
    symbols: Vec<u16>,
}

impl Huffman {
    fn new(lengths: &[u8]) -> Self {
        let mut counts = [0u16; 16];
        for &l in lengths {
            counts[l as usize] += 1;
        }
        counts[0] = 0;
        let mut offsets = [0u16; 16];
        for len in 1..15 {
            offsets[len + 1] = offsets[len] + counts[len];
        }
        let mut symbols = vec![0u16; lengths.len()];
        for (symbol, &l) in (0u16..).zip(lengths) {
            if l != 0 {
                symbols[usize::from(offsets[usize::from(l)])] = symbol;
                offsets[usize::from(l)] += 1;
            }
        }
        Self { counts, symbols }
    }

    fn decode(&self, bits: &mut Bits<'_>) -> Result<u16> {
        let (mut code, mut first, mut index) = (0usize, 0usize, 0usize);
        for len in 1..16 {
            code |= usize::from(bits.take(1)? == 1);
            let count = usize::from(self.counts[len]);
            // Codes of this length are first..first+count; a code below `first` is a longer one's prefix.
            if code >= first && code - first < count {
                return Ok(self.symbols[index + code - first]);
            }
            index += count;
            first = (first + count) << 1;
            code <<= 1;
        }
        Err(Error::malformed_at(
            bits.pos,
            "invalid Huffman code in deflate data",
        ))
    }
}

/// Decompress `data`: a zlib stream, or bare DEFLATE when no zlib header is there.
pub fn inflate(data: &[u8]) -> Result<Vec<u8>> {
    let zlib = data.len() >= 2
        && data[0] & 0x0f == 8
        && (u16::from(data[0]) << 8 | u16::from(data[1])) % 31 == 0;
    let mut bits = Bits {
        data,
        pos: if zlib { 2 } else { 0 },
        buf: 0,
        count: 0,
    };
    let mut out = Vec::with_capacity(data.len() * 4);
    loop {
        let last = bits.take(1)? == 1;
        match bits.take(2)? {
            0 => stored(&mut bits, &mut out)?,
            1 => {
                let (lit, dist) = fixed();
                codes(&mut bits, &mut out, &lit, &dist)?;
            }
            2 => {
                let (lit, dist) = dynamic(&mut bits)?;
                codes(&mut bits, &mut out, &lit, &dist)?;
            }
            _ => return Err(Error::malformed_at(bits.pos, "invalid deflate block type")),
        }
        if last {
            return Ok(out);
        }
    }
}

fn stored(bits: &mut Bits<'_>, out: &mut Vec<u8>) -> Result<()> {
    bits.align();
    let at = bits.pos;
    let head = bits
        .data
        .get(at..at + 4)
        .ok_or_else(|| Error::malformed_at(at, "stored block header cut short"))?;
    let len16 = u16::from_le_bytes([head[0], head[1]]);
    let nlen = u16::from_le_bytes([head[2], head[3]]);
    let len = usize::from(len16);
    if nlen != !len16 {
        return Err(Error::malformed_at(at, "stored block length check fails"));
    }
    let body = bits
        .data
        .get(at + 4..at + 4 + len)
        .ok_or_else(|| Error::malformed_at(at, "stored block cut short"))?;
    out.extend_from_slice(body);
    bits.pos = at + 4 + len;
    Ok(())
}

fn fixed() -> (Huffman, Huffman) {
    let mut lengths = [0u8; 288];
    lengths[..144].fill(8);
    lengths[144..256].fill(9);
    lengths[256..280].fill(7);
    lengths[280..].fill(8);
    (Huffman::new(&lengths), Huffman::new(&[5u8; 30]))
}

fn dynamic(bits: &mut Bits<'_>) -> Result<(Huffman, Huffman)> {
    let nlen = bits.take(5)? as usize + 257;
    let ndist = bits.take(5)? as usize + 1;
    let ncode = bits.take(4)? as usize + 4;
    if nlen > 286 || ndist > 30 {
        return Err(Error::malformed_at(
            bits.pos,
            "deflate code counts out of range",
        ));
    }
    let mut code_lengths = [0u8; 19];
    for &slot in &CODE_LENGTH_ORDER[..ncode] {
        code_lengths[slot] = bits.take(3)?.to_le_bytes()[0];
    }
    let code_lengths = Huffman::new(&code_lengths);
    let mut lengths = vec![0u8; nlen + ndist];
    let mut i = 0;
    while i < lengths.len() {
        let symbol = code_lengths.decode(bits)?;
        let (value, repeat) = match symbol {
            0..=15 => (symbol.to_le_bytes()[0], 1),
            16 => {
                let prev = *lengths[..i].last().ok_or_else(|| {
                    Error::malformed_at(bits.pos, "length repeat with no previous")
                })?;
                (prev, 3 + bits.take(2)? as usize)
            }
            17 => (0, 3 + bits.take(3)? as usize),
            _ => (0, 11 + bits.take(7)? as usize),
        };
        if i + repeat > lengths.len() {
            return Err(Error::malformed_at(
                bits.pos,
                "deflate code lengths overrun",
            ));
        }
        lengths[i..i + repeat].fill(value);
        i += repeat;
    }
    Ok((
        Huffman::new(&lengths[..nlen]),
        Huffman::new(&lengths[nlen..]),
    ))
}

fn codes(bits: &mut Bits<'_>, out: &mut Vec<u8>, lit: &Huffman, dist: &Huffman) -> Result<()> {
    loop {
        let symbol = lit.decode(bits)?;
        match symbol {
            0..=255 => out.push(symbol.to_le_bytes()[0]),
            256 => return Ok(()),
            _ => {
                let s = usize::from(symbol - 257);
                if s >= 29 {
                    return Err(Error::malformed_at(bits.pos, "invalid deflate length code"));
                }
                let len =
                    usize::from(LENGTH_BASE[s]) + bits.take(u32::from(LENGTH_EXTRA[s]))? as usize;
                let d = usize::from(dist.decode(bits)?);
                if d >= 30 {
                    return Err(Error::malformed_at(
                        bits.pos,
                        "invalid deflate distance code",
                    ));
                }
                let back =
                    usize::from(DIST_BASE[d]) + bits.take(u32::from(DIST_EXTRA[d]))? as usize;
                if back > out.len() {
                    return Err(Error::malformed_at(
                        bits.pos,
                        "deflate distance reaches before the start",
                    ));
                }
                let from = out.len() - back;
                for k in 0..len {
                    out.push(out[from + k]);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(text: &str) -> Vec<u8> {
        (0..text.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&text[i..i + 2], 16).unwrap())
            .collect()
    }

    // Vectors made with Python's zlib.compress (levels 6, 0 and 9).
    #[test]
    fn a_fixed_huffman_block_inflates() {
        assert_eq!(
            inflate(&hex("789ccb48cdc9c957c84090003a2e067d")).unwrap(),
            b"hello hello hello"
        );
    }

    #[test]
    fn a_stored_block_inflates() {
        assert_eq!(
            inflate(&hex("7801010d00f2ff73746f72656420626c6f636b21245e04de")).unwrap(),
            b"stored block!"
        );
    }

    #[test]
    fn a_dynamic_huffman_block_inflates() {
        let expected: String = (0..300u64).map(|i| (i * i).to_string()).collect();
        let packed = include_str!("inflate_dynamic.hex").trim();
        assert_eq!(inflate(&hex(packed)).unwrap(), expected.as_bytes());
    }

    #[test]
    fn bare_deflate_without_the_zlib_header_inflates() {
        let zlib = hex("789ccb48cdc9c957c84090003a2e067d");
        assert_eq!(
            inflate(&zlib[2..zlib.len() - 4]).unwrap(),
            b"hello hello hello"
        );
    }

    #[test]
    fn truncated_data_is_refused() {
        let zlib = hex("789ccb48cdc9c957c84090003a2e067d");
        assert!(inflate(&zlib[..6]).is_err());
    }
}
