//! `CMap`s (ISO 32000-2 sections 9.7.5 and 9.10.3): how a string splits into character codes, which `CID` a code
//! selects, and which Unicode text a code stands for.

use std::collections::HashMap;

use crate::content::operations;
use crate::object::Object;

/// A `CMap`: its code space and its mappings, to `CID`s (an encoding `CMap`) or to text (a `ToUnicode` `CMap`).
#[derive(Debug, Clone, Default)]
pub struct CMap {
    /// Code space ranges: (low, high), each as bytes of one length.
    ranges: Vec<(Vec<u8>, Vec<u8>)>,
    /// Single codes to `CID`s.
    cids: HashMap<u32, u32>,
    /// Code ranges to their first `CID`: (low, high, first).
    cid_ranges: Vec<(u32, u32, u32)>,
    /// Codes to Unicode text.
    text: HashMap<u32, String>,
    /// The `CMap` is Identity: every 2-byte code is its own `CID`.
    identity: bool,
    /// The writing mode its program defines (`/WMode 1 def`: vertical), if it defines one.
    wmode: Option<i64>,
}

impl CMap {
    /// `Identity-H` / `Identity-V`: two-byte codes, each its own `CID`.
    #[must_use]
    pub fn identity() -> Self {
        Self {
            ranges: vec![(vec![0, 0], vec![0xFF, 0xFF])],
            identity: true,
            ..Self::default()
        }
    }

    /// A predefined `CMap` of a legacy CJK encoding (`GBK-EUC-H`, `B5pc-H`, `90ms-RKSJ-H`, `KSC-EUC-H`, ...): its
    /// code space, the CIDs of its one-byte ASCII codes (1-95 in every Adobe CJK collection), and the encoding its
    /// codes are bytes of. None for a name that is not one.
    #[must_use]
    pub fn legacy(name: &str) -> Option<(Self, &'static encoding_rs::Encoding)> {
        let base = name.trim_end_matches("-H").trim_end_matches("-V");
        let (encoding, doubles): (&'static encoding_rs::Encoding, &[(u8, u8)]) = match base {
            "GB-EUC" | "GBpc-EUC" | "GBK-EUC" | "GBKp-EUC" | "GBK2K" => {
                (encoding_rs::GBK, &[(0x81, 0xFE)])
            }
            "B5pc" | "ETen-B5" | "HKscs-B5" | "B5" | "ETenms-B5" => {
                (encoding_rs::BIG5, &[(0xA1, 0xFE)])
            }
            "90ms-RKSJ" | "90msp-RKSJ" | "83pv-RKSJ" | "90pv-RKSJ" | "Add-RKSJ" | "Ext-RKSJ" => {
                (encoding_rs::SHIFT_JIS, &[(0x81, 0x9F), (0xE0, 0xFC)])
            }
            "EUC" => (encoding_rs::EUC_JP, &[(0x8E, 0x8E), (0xA1, 0xFE)]),
            "KSC-EUC" | "KSCms-UHC" | "KSCpc-EUC" | "KSCms-UHC-HW" => {
                (encoding_rs::EUC_KR, &[(0x81, 0xFE)])
            }
            _ => return None,
        };
        let mut ranges = vec![(vec![0x00], vec![0x80])];
        for &(lo, hi) in doubles {
            ranges.push((vec![lo, 0x40], vec![hi, 0xFE]));
        }
        if encoding == encoding_rs::SHIFT_JIS {
            ranges.push((vec![0xA0], vec![0xDF])); // half-width katakana
        }
        let cmap = Self {
            ranges,
            cid_ranges: vec![(0x20, 0x7E, 1)],
            ..Self::default()
        };
        Some((cmap, encoding))
    }

    /// The `CMap` a stream's data defines (`usecmap` is not followed: a parent `CMap` is named, not embedded).
    #[must_use]
    pub fn parse(data: &[u8]) -> Self {
        let mut cmap = Self::default();
        let (ops, _) = operations(data);
        for op in ops {
            let items = &op.operands;
            match op.operator.as_slice() {
                b"endcodespacerange" => {
                    for pair in items.as_chunks::<2>().0 {
                        if let (Object::String(lo), Object::String(hi)) = (&pair[0], &pair[1])
                            && lo.len() == hi.len()
                            && !lo.is_empty()
                        {
                            cmap.ranges.push((lo.clone(), hi.clone()));
                        }
                    }
                }
                b"endbfchar" => {
                    for pair in items.as_chunks::<2>().0 {
                        if let (Object::String(code), Some(text)) = (&pair[0], unicode(&pair[1])) {
                            cmap.text.insert(value(code), text);
                        }
                    }
                }
                b"endbfrange" => {
                    for triple in items.as_chunks::<3>().0 {
                        cmap.bf_range(triple);
                    }
                }
                b"endcidchar" => {
                    for pair in items.as_chunks::<2>().0 {
                        if let (Object::String(code), Some(cid)) = (&pair[0], pair[1].as_integer())
                        {
                            cmap.cids
                                .insert(value(code), u32::try_from(cid).unwrap_or(0));
                        }
                    }
                }
                b"def" => {
                    if let [.., Object::Name(key), Object::Integer(mode)] = items.as_slice()
                        && key == b"WMode"
                    {
                        cmap.wmode = Some(*mode);
                    }
                }
                b"endcidrange" => {
                    for triple in items.as_chunks::<3>().0 {
                        if let (Object::String(lo), Object::String(hi), Some(cid)) =
                            (&triple[0], &triple[1], triple[2].as_integer())
                        {
                            cmap.cid_ranges.push((
                                value(lo),
                                value(hi),
                                u32::try_from(cid).unwrap_or(0),
                            ));
                        }
                    }
                }
                _ => {}
            }
        }
        cmap
    }

    /// The writing mode the `CMap`'s program defines (0 horizontal, 1 vertical), if it defines one.
    #[must_use]
    pub fn wmode(&self) -> Option<i64> {
        self.wmode
    }

    fn bf_range(&mut self, triple: &[Object]) {
        let (Object::String(lo), Object::String(hi)) = (&triple[0], &triple[1]) else {
            return;
        };
        let (lo, hi) = (value(lo), value(hi));
        // A range past 2^16 codes is a hostile or broken file, not text.
        if hi < lo || hi - lo > 0xFFFF {
            return;
        }
        match &triple[2] {
            Object::String(start) => {
                let Some(first) = utf16(start) else { return };
                let mut units: Vec<char> = first.chars().collect();
                for code in lo..=hi {
                    self.text.insert(code, units.iter().collect());
                    // The last character increments with the code.
                    if let Some(last) = units.last_mut() {
                        *last = char::from_u32(u32::from(*last) + 1).unwrap_or(*last);
                    }
                }
            }
            Object::Array(texts) => {
                for (code, text) in (lo..=hi).zip(texts) {
                    if let Some(t) = unicode(text) {
                        self.text.insert(code, t);
                    }
                }
            }
            _ => {}
        }
    }

    /// Split `bytes` into codes by the code space: (code, its byte length). A byte that starts no code in the
    /// space is a one-byte code (as a viewer reads it).
    #[must_use]
    pub fn codes(&self, bytes: &[u8]) -> Vec<(u32, usize)> {
        let mut out = Vec::new();
        let mut i = 0;
        while i < bytes.len() {
            let len = (1..=4)
                .find(|&n| {
                    bytes.get(i..i + n).is_some_and(|c| {
                        self.ranges.iter().any(|(lo, hi)| {
                            lo.len() == n
                                && c.iter().zip(lo).zip(hi).all(|((b, l), h)| l <= b && b <= h)
                        })
                    })
                })
                .unwrap_or(if self.ranges.is_empty() {
                    1
                } else {
                    self.shortest()
                });
            let len = len.min(bytes.len() - i);
            out.push((value(&bytes[i..i + len]), len));
            i += len;
        }
        out
    }

    fn shortest(&self) -> usize {
        self.ranges
            .iter()
            .map(|(lo, _)| lo.len())
            .min()
            .unwrap_or(1)
    }

    /// The `CID` `code` selects (0, `.notdef`, when the `CMap` maps it to none).
    #[must_use]
    pub fn cid(&self, code: u32) -> u32 {
        if self.identity {
            return code;
        }
        if let Some(&cid) = self.cids.get(&code) {
            return cid;
        }
        self.cid_ranges
            .iter()
            .find(|(lo, hi, _)| (*lo..=*hi).contains(&code))
            .map_or(0, |(lo, _, first)| first + code - lo)
    }

    /// The text `code` stands for, when this is a `ToUnicode` `CMap` that maps it.
    #[must_use]
    pub fn text(&self, code: u32) -> Option<&str> {
        self.text.get(&code).map(String::as_str)
    }

    /// Whether the code space is a single byte wide everywhere.
    #[must_use]
    pub fn single_byte(&self) -> bool {
        self.ranges.iter().all(|(lo, _)| lo.len() == 1)
    }
}

/// A code's bytes as a big-endian number.
fn value(bytes: &[u8]) -> u32 {
    bytes
        .iter()
        .take(4)
        .fold(0u32, |acc, &b| acc << 8 | u32::from(b))
}

/// A `ToUnicode` destination: UTF-16BE text, or a name (rare, from old writers).
fn unicode(object: &Object) -> Option<String> {
    match object {
        Object::String(s) => utf16(s),
        Object::Name(n) => {
            pdf_encoding::glyphname_to_unicode(&String::from_utf8_lossy(n)).map(str::to_string)
        }
        _ => None,
    }
}

/// UTF-16BE bytes as text (one odd trailing byte is read as its own code unit).
fn utf16(bytes: &[u8]) -> Option<String> {
    let units: Vec<u16> = bytes
        .chunks(2)
        .map(|c| {
            if c.len() == 2 {
                u16::from_be_bytes([c[0], c[1]])
            } else {
                u16::from(c[0])
            }
        })
        .collect();
    let text: String = char::decode_utf16(units)
        .map(|r| r.unwrap_or('\u{FFFD}'))
        .collect();
    (!text.is_empty()).then_some(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TO_UNICODE: &[u8] = b"/CIDInit /ProcSet findresource begin 12 dict begin begincmap
1 begincodespacerange <0000> <FFFF> endcodespacerange
2 beginbfchar <0003> <0020> <0024> <00660069> endbfchar
2 beginbfrange <0010> <0012> <0041> <0020> <0021> [<03B1> <D835DF0F>] endbfrange
endcmap CMapName currentdict /CMap defineresource pop end end";

    #[test]
    fn a_to_unicode_cmap_maps_chars_ranges_and_ligatures() {
        let cmap = CMap::parse(TO_UNICODE);
        assert_eq!(cmap.text(0x03), Some(" "));
        assert_eq!(cmap.text(0x24), Some("fi"));
        assert_eq!(cmap.text(0x11), Some("B"));
        assert_eq!(cmap.text(0x20), Some("\u{3B1}"));
        assert_eq!(cmap.text(0x21), Some("\u{1D70F}"));
        assert_eq!(cmap.codes(&[0, 3, 0, 0x24]), vec![(3, 2), (0x24, 2)]);
    }

    #[test]
    fn a_mixed_width_code_space_splits_by_its_ranges() {
        let cmap = CMap::parse(b"2 begincodespacerange <00> <80> <8140> <9FFC> endcodespacerange");
        assert_eq!(
            cmap.codes(&[0x41, 0x81, 0x40, 0x42]),
            vec![(0x41, 1), (0x8140, 2), (0x42, 1)]
        );
    }

    #[test]
    fn cid_ranges_and_identity_select_cids() {
        let cmap = CMap::parse(b"1 begincodespacerange <0000> <FFFF> endcodespacerange 1 begincidrange <0100> <01FF> 500 endcidrange");
        assert_eq!((cmap.cid(0x0105), cmap.cid(0x0001)), (505, 0));
        assert_eq!(CMap::identity().cid(0x1234), 0x1234);
    }
    #[test]
    fn a_legacy_cjk_cmap_splits_its_encoding_and_numbers_its_ascii() {
        let (cmap, encoding) = CMap::legacy("GBK-EUC-H").unwrap();
        assert_eq!(encoding, encoding_rs::GBK);
        assert_eq!(
            cmap.codes(&[0x41, 0xC4, 0xE3]),
            vec![(0x41, 1), (0xC4E3, 2)]
        );
        assert_eq!(cmap.cid(0x41), 0x22);
        assert!(CMap::legacy("Adobe-Japan1-6").is_none());
    }
}
