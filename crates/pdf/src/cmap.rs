//! `CMap`s (ISO 32000-2 sections 9.7.5 and 9.10.3), embedded or predefined: how a string splits into character
//! codes, which `CID` a code selects, which Unicode text a code stands for, and which text a `CID` of an Adobe
//! character collection stands for.

use std::collections::HashMap;

use crate::cid2unicode::{COLLECTIONS, Collection};
use crate::cmaps::{CIDS, PREDEFINED, Predefined, SEGMENTS};
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
    /// The predefined `CMap` this is, whose tables give the `CID`s.
    predefined: Option<&'static Predefined>,
    /// The writing mode an embedded `CMap`'s program defines (`/WMode 1 def`: vertical), if it defines one.
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

    /// A predefined `CMap` (ISO 32000-2 Table 116: `GBK-EUC-H`, `90ms-RKSJ-V`, `UniJIS-UTF16-H`, ...), as Adobe
    /// defines it: the `CMap`, whether its writing mode is vertical, and the text table of its collection. None for a
    /// name that is not one (`Identity-H` and `-V` are [`CMap::identity`]).
    #[must_use]
    pub fn predefined(name: &str) -> Option<(Self, bool, Option<&'static Collection>)> {
        let table = PREDEFINED.iter().find(|p| p.name == name)?;
        let cmap = Self {
            ranges: table
                .space
                .iter()
                .map(|(lo, hi)| (lo.to_vec(), hi.to_vec()))
                .collect(),
            predefined: Some(table),
            ..Self::default()
        };
        let collection = Collection::named(b"Adobe", table.ordering.as_bytes());
        Some((cmap, table.vertical, collection))
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
        if let Some(table) = self.predefined {
            return predefined_cid(table, code);
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

impl Collection {
    /// Adobe's table of the character collection a `/CIDSystemInfo` names -- `/Registry (Adobe)` and an `/Ordering`
    /// it covers (`Japan1`, `GB1`, `CNS1`, `Korea1`, `KR`): the CID-to-Unicode `CMap` `Adobe-<Ordering>-UCS2` that
    /// ISO 32000-2 section 9.10.2 names. None for any other collection.
    #[must_use]
    pub fn named(registry: &[u8], ordering: &[u8]) -> Option<&'static Self> {
        if registry != b"Adobe" {
            return None;
        }
        COLLECTIONS
            .iter()
            .find(|c| c.ordering.as_bytes() == ordering)
    }

    /// The text of `cid`, when the table gives it one.
    #[must_use]
    pub fn text(&self, cid: u32) -> Option<String> {
        let at = usize::try_from(cid).ok()?.checked_mul(2)?;
        let unit = self
            .single
            .get(at..at + 2)
            .map_or(0, |b| u16::from_le_bytes([b[0], b[1]]));
        if unit != 0 {
            return char::from_u32(u32::from(unit)).map(String::from);
        }
        let cid = u16::try_from(cid).ok()?;
        self.long
            .binary_search_by_key(&cid, |&(c, _)| c)
            .ok()
            .map(|i| self.long[i].1.to_string())
    }
}

/// The `CID` a predefined `CMap` selects for `code`: its own segment's, else its base's (0 when none maps it).
fn predefined_cid(mut table: &'static Predefined, code: u32) -> u32 {
    let word = |segment: usize, field: usize| {
        let at = segment * 12 + field * 4;
        u32::from_le_bytes([
            SEGMENTS[at],
            SEGMENTS[at + 1],
            SEGMENTS[at + 2],
            SEGMENTS[at + 3],
        ])
    };
    loop {
        // The segments are sorted and disjoint: the last one starting at or below `code` is the only candidate.
        let (mut low, mut high) = (table.first, table.first + table.count);
        while low < high {
            let middle = usize::midpoint(low, high);
            if word(middle, 0) <= code {
                low = middle + 1;
            } else {
                high = middle;
            }
        }
        if low > table.first && code <= word(low - 1, 1) {
            let (first, value) = (word(low - 1, 0), word(low - 1, 2));
            let offset = code - first;
            return if value & DENSE == 0 {
                if value == 0 { 0 } else { value + offset }
            } else {
                let at = 2 * usize::try_from((value & !DENSE) + offset).unwrap_or(usize::MAX);
                CIDS.get(at..at + 2)
                    .map_or(0, |b| u32::from(u16::from_le_bytes([b[0], b[1]])))
            };
        }
        match table.base {
            Some(base) => table = &PREDEFINED[base],
            None => return 0,
        }
    }
}

/// The flag of a dense segment's value in [`SEGMENTS`]: the rest is the index of its first `CID` in [`CIDS`].
const DENSE: u32 = 1 << 31;

/// The encoding whose bytes a predefined `CMap`'s codes are, for the legacy CJK ones (`GBK-EUC-H`, `B5pc-H`,
/// `90ms-RKSJ-H`, `KSC-EUC-H`, ...): their text where no collection table gives it. None for any other.
#[must_use]
pub fn legacy_encoding(name: &str) -> Option<&'static encoding_rs::Encoding> {
    let base = name.trim_end_matches("-H").trim_end_matches("-V");
    Some(match base {
        "GB-EUC" | "GBpc-EUC" | "GBK-EUC" | "GBKp-EUC" | "GBK2K" => encoding_rs::GBK,
        "B5pc" | "ETen-B5" | "HKscs-B5" | "ETenms-B5" => encoding_rs::BIG5,
        "90ms-RKSJ" | "90msp-RKSJ" | "83pv-RKSJ" | "90pv-RKSJ" | "Add-RKSJ" | "Ext-RKSJ" => {
            encoding_rs::SHIFT_JIS
        }
        "EUC" => encoding_rs::EUC_JP,
        "KSC-EUC" | "KSCms-UHC" | "KSCpc-EUC" | "KSCms-UHC-HW" => encoding_rs::EUC_KR,
        _ => return None,
    })
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
    /// A predefined `CMap`'s `CID`s of `codes`.
    fn predefined_cids(name: &str, codes: &[u32]) -> Vec<u32> {
        let (cmap, ..) = CMap::predefined(name).unwrap();
        codes.iter().map(|&c| cmap.cid(c)).collect()
    }

    #[test]
    fn predefined_cmaps_select_the_cids_adobe_states() {
        // Each code's CID as Adobe's cid2code.txt states it (cmap-resources 20231115): one-byte codes are not 1-95.
        for (name, codes, cids) in [
            (
                "GBK-EUC-H",
                &[0x20, 0x21, 0xC4E3][..],
                &[7716, 814, 2886][..],
            ),
            ("GB-EUC-H", &[0x41, 0xD6D0], &[846, 4559]),
            ("B5pc-H", &[0x41, 0xA440], &[34, 595]),
            ("90ms-RKSJ-H", &[0x41, 0x82A0, 0x8141], &[264, 843, 634]),
            ("EUC-H", &[0xA4A2], &[843]),
            ("H", &[0x2422, 0x2122], &[843, 634]),
            ("KSC-EUC-H", &[0x41, 0xB0A1], &[8127, 1086]),
            ("UniJIS-UTF16-H", &[0x3042, 0xD840_DC0B], &[843, 13839]),
            ("UniGB-UCS2-H", &[0x4E2D], &[4559]),
        ] {
            assert_eq!(predefined_cids(name, codes), cids, "{name}");
        }
    }

    #[test]
    fn a_vertical_cmap_overrides_its_horizontal_base() {
        // The ideographic comma's vertical form; every other code as the horizontal `CMap` has it.
        assert_eq!(
            predefined_cids("90ms-RKSJ-V", &[0x8141, 0x82A0]),
            [7887, 843]
        );
        assert_eq!(predefined_cids("V", &[0x2122]), [7887]);
        let (_, vertical, collection) = CMap::predefined("90ms-RKSJ-V").unwrap();
        assert!(vertical);
        assert_eq!(collection.map(|c| c.ordering), Some("Japan1"));
        assert!(!CMap::predefined("GBK-EUC-H").unwrap().1);
    }

    #[test]
    fn a_predefined_cmap_splits_by_its_code_space() {
        let (cmap, ..) = CMap::predefined("GBK-EUC-H").unwrap();
        assert_eq!(
            cmap.codes(&[0x41, 0xC4, 0xE3]),
            vec![(0x41, 1), (0xC4E3, 2)]
        );
        let (cmap, ..) = CMap::predefined("UniJIS-UTF16-H").unwrap();
        assert_eq!(
            cmap.codes(&[0x30, 0x42, 0xD8, 0x40, 0xDC, 0x0B]),
            vec![(0x3042, 2), (0xD840_DC0B, 4)]
        );
        assert!(CMap::predefined("Identity-H").is_none());
        assert!(CMap::predefined("Adobe-Japan1-6").is_none());
    }

    #[test]
    fn a_legacy_cmap_names_the_encoding_of_its_bytes() {
        assert_eq!(legacy_encoding("GBK-EUC-V"), Some(encoding_rs::GBK));
        assert_eq!(legacy_encoding("90ms-RKSJ-H"), Some(encoding_rs::SHIFT_JIS));
        assert_eq!(legacy_encoding("H"), None);
    }
}
