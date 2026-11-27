//! DEFLATE (RFC 1951) inside an optional zlib wrapper (RFC 1950): what `/FlateDecode` holds. <!-- spelling-ignore-line: the PDF name -->

use std::sync::OnceLock;

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

/// How many bits a code's first look-up reads: a code this long or shorter is found in one step, a longer one bit by
/// bit (rare: a block's codes are mostly short).
const FAST_BITS: u32 = 10;

/// The input as bits, least significant first: a 64-bit buffer filled a word at a time.
struct Bits<'a> {
    data: &'a [u8],
    /// The next byte not yet in the buffer.
    next: usize,
    buf: u64,
    /// How many bits of the buffer are unread.
    count: u32,
}

impl Bits<'_> {
    /// Fill the buffer to 56 bits or more, or with all the input left.
    fn refill(&mut self) {
        if let Some(word) = self.data.get(self.next..self.next + 8) {
            let word = u64::from_le_bytes(word.try_into().expect("eight bytes"));
            // The bits past the whole bytes counted are ORed in again, in the same place, by the next fill.
            self.buf |= word << self.count;
            self.next += ((63 - self.count) >> 3) as usize;
            self.count |= 0b11_1000; // 56 or more: the whole bytes added
        } else {
            while self.count <= 56 && self.next < self.data.len() {
                self.buf |= u64::from(self.data[self.next]) << self.count;
                self.next += 1;
                self.count += 8;
            }
        }
    }

    /// Where the reading stands, as a byte offset: past the byte holding the last bit read.
    fn pos(&self) -> usize {
        self.next - (self.count >> 3) as usize
    }

    /// The next `n` bits (at most 32), not yet read; fewer than `n` are there when the input ends first.
    fn peek(&mut self, n: u32) -> u32 {
        if self.count < n {
            self.refill();
        }
        u32::try_from(self.buf & ((1u64 << n) - 1)).expect("at most 32 bits")
    }

    fn consume(&mut self, n: u32) {
        self.buf >>= n;
        self.count -= n;
    }

    /// The next `n` bits, the buffer holding them already.
    #[inline]
    fn take_filled(&mut self, n: u32) -> usize {
        debug_assert!(n <= self.count);
        let v = self.buf & ((1u64 << n) - 1);
        self.consume(n);
        usize::try_from(v).expect("a few bits")
    }

    fn take(&mut self, n: u32) -> Result<u32> {
        let v = self.peek(n);
        if self.count < n {
            return Err(Error::malformed_at(
                self.data.len(),
                "deflate data ends early",
            ));
        }
        self.consume(n);
        Ok(v)
    }

    /// Skip to the next byte boundary.
    fn align(&mut self) {
        self.next -= (self.count >> 3) as usize;
        self.buf = 0;
        self.count = 0;
    }
}

/// A canonical Huffman code: how many codes of each length and the symbols in code order, and the symbol of every
/// code of up to `bits` bits by those bits (as read, least significant first).
struct Huffman {
    counts: [u16; 16],
    symbols: Vec<u16>,
    /// The bits the first look-up reads: [`FAST_BITS`], or the longest code's length when shorter.
    bits: u32,
    /// By the next `bits` bits: the symbol shifted left 4, or-ed with its code's length; 0 where no code of up to
    /// `bits` bits begins those bits.
    fast: Vec<u16>,
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
        // Each code in canonical order (by length, then symbol), filled in wherever its bits begin what is read;
        // the first code filled in keeps a slot. A code that does not fit its length (an over-full set) is left to
        // the bit-by-bit decoding, as a longer code is.
        let longest = (1..16u32)
            .rev()
            .find(|&l| counts[l as usize] != 0)
            .unwrap_or(1);
        let bits = longest.min(FAST_BITS);
        let mut fast = vec![0u16; 1 << bits];
        let (mut code, mut index) = (0u32, 0usize);
        for len in 1..=bits {
            let count = usize::from(counts[len as usize]);
            for &symbol in &symbols[index..index + count] {
                if code < 1 << len {
                    let reversed = code.reverse_bits() >> (32 - len);
                    let entry = symbol << 4 | u16::try_from(len).expect("a short length");
                    for slot in (reversed as usize..fast.len()).step_by(1 << len) {
                        if fast[slot] == 0 {
                            fast[slot] = entry;
                        }
                    }
                }
                code += 1;
            }
            index += count;
            code <<= 1;
        }
        Self {
            counts,
            symbols,
            bits,
            fast,
        }
    }

    #[inline]
    fn decode(&self, bits: &mut Bits<'_>) -> Result<u16> {
        let entry = self.fast[bits.peek(self.bits) as usize];
        let len = u32::from(entry & 0xf);
        if len != 0 && len <= bits.count {
            bits.consume(len);
            return Ok(entry >> 4);
        }
        self.decode_bitwise(bits)
    }

    /// The next symbol, the buffer holding [`FAST_BITS`] bits or more.
    #[inline]
    fn decode_filled(&self, bits: &mut Bits<'_>) -> Result<u16> {
        let index = usize::try_from(bits.buf & ((1u64 << self.bits) - 1)).expect("a few bits");
        let entry = self.fast[index];
        if entry == 0 {
            return self.decode_bitwise(bits);
        }
        bits.consume(u32::from(entry & 0xf));
        Ok(entry >> 4)
    }

    /// The canonical decoding, a bit at a time: for a code longer than the first look-up reads, and at the end of
    /// the input.
    #[cold]
    #[inline(never)]
    fn decode_bitwise(&self, bits: &mut Bits<'_>) -> Result<u16> {
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
            bits.pos(),
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
        next: if zlib { 2 } else { 0 },
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
                codes(&mut bits, &mut out, lit, dist)?;
            }
            2 => {
                let (lit, dist) = dynamic(&mut bits)?;
                codes(&mut bits, &mut out, &lit, &dist)?;
            }
            _ => {
                return Err(Error::malformed_at(
                    bits.pos(),
                    "invalid deflate block type",
                ));
            }
        }
        if last {
            return Ok(out);
        }
    }
}

fn stored(bits: &mut Bits<'_>, out: &mut Vec<u8>) -> Result<()> {
    bits.align();
    let at = bits.next;
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
    bits.next = at + 4 + len;
    Ok(())
}

/// The fixed codes (3.2.6), made once.
fn fixed() -> &'static (Huffman, Huffman) {
    static FIXED: OnceLock<(Huffman, Huffman)> = OnceLock::new();
    FIXED.get_or_init(|| {
        let mut lengths = [0u8; 288];
        lengths[..144].fill(8);
        lengths[144..256].fill(9);
        lengths[256..280].fill(7);
        lengths[280..].fill(8);
        (Huffman::new(&lengths), Huffman::new(&[5u8; 30]))
    })
}

fn dynamic(bits: &mut Bits<'_>) -> Result<(Huffman, Huffman)> {
    let nlen = bits.take(5)? as usize + 257;
    let ndist = bits.take(5)? as usize + 1;
    let ncode = bits.take(4)? as usize + 4;
    if nlen > 286 || ndist > 30 {
        return Err(Error::malformed_at(
            bits.pos(),
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
                    Error::malformed_at(bits.pos(), "length repeat with no previous")
                })?;
                (prev, 3 + bits.take(2)? as usize)
            }
            17 => (0, 3 + bits.take(3)? as usize),
            _ => (0, 11 + bits.take(7)? as usize),
        };
        if i + repeat > lengths.len() {
            return Err(Error::malformed_at(
                bits.pos(),
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

/// A block's codes into `out`, up to its end code.
fn codes(bits: &mut Bits<'_>, out: &mut Vec<u8>, lit: &Huffman, dist: &Huffman) -> Result<()> {
    // While eight bytes of input remain, one fill holds 56 bits or more: all a literal, or a length and distance
    // with their extra bits, can take (48 at most), so none of the reads below runs dry.
    while bits.next + 8 <= bits.data.len() {
        bits.refill();
        let symbol = lit.decode_filled(bits)?;
        if symbol < 256 {
            out.push(symbol.to_le_bytes()[0]);
            continue;
        }
        if symbol == 256 {
            return Ok(());
        }
        let s = usize::from(symbol - 257);
        if s >= 29 {
            return Err(Error::malformed_at(
                bits.pos(),
                "invalid deflate length code",
            ));
        }
        let len = usize::from(LENGTH_BASE[s]) + bits.take_filled(u32::from(LENGTH_EXTRA[s]));
        let d = usize::from(dist.decode_filled(bits)?);
        if d >= 30 {
            return Err(Error::malformed_at(
                bits.pos(),
                "invalid deflate distance code",
            ));
        }
        let back = usize::from(DIST_BASE[d]) + bits.take_filled(u32::from(DIST_EXTRA[d]));
        copy(bits, out, back, len)?;
    }
    // The last bytes: every read checked.
    loop {
        let symbol = lit.decode(bits)?;
        match symbol {
            0..=255 => out.push(symbol.to_le_bytes()[0]),
            256 => return Ok(()),
            _ => {
                let s = usize::from(symbol - 257);
                if s >= 29 {
                    return Err(Error::malformed_at(
                        bits.pos(),
                        "invalid deflate length code",
                    ));
                }
                let len =
                    usize::from(LENGTH_BASE[s]) + bits.take(u32::from(LENGTH_EXTRA[s]))? as usize;
                let d = usize::from(dist.decode(bits)?);
                if d >= 30 {
                    return Err(Error::malformed_at(
                        bits.pos(),
                        "invalid deflate distance code",
                    ));
                }
                let back =
                    usize::from(DIST_BASE[d]) + bits.take(u32::from(DIST_EXTRA[d]))? as usize;
                copy(bits, out, back, len)?;
            }
        }
    }
}

/// Append the `len` bytes that begin `back` bytes before the end of `out`.
fn copy(bits: &Bits<'_>, out: &mut Vec<u8>, back: usize, len: usize) -> Result<()> {
    if back > out.len() {
        return Err(Error::malformed_at(
            bits.pos(),
            "deflate distance reaches before the start",
        ));
    }
    let from = out.len() - back;
    if back >= len {
        out.extend_from_within(from..from + len);
    } else if back == 1 {
        out.resize(out.len() + len, out[from]); // a run of one byte
    } else {
        // The copy overlaps what it writes: a run repeating the last `back` bytes.
        for k in 0..len {
            out.push(out[from + k]);
        }
    }
    Ok(())
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
    fn codes_longer_than_the_first_look_up_inflate() {
        // Letters of falling frequency (a half, a quarter, ...) from a fixed LCG: literal codes up to 12 bits.
        // Made with Python's zlib.compress at level 9.
        let mut x = 1u64;
        let expected: Vec<u8> = (0..16384)
            .map(|_| {
                x = (x * 1_103_515_245 + 12345) % (1 << 31);
                let v = (x >> 16) | (1 << 15);
                b'a' + u8::try_from(v.trailing_zeros()).unwrap()
            })
            .collect();
        let packed = include_str!("inflate_long_codes.hex").trim();
        assert_eq!(inflate(&hex(packed)).unwrap(), expected);
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
