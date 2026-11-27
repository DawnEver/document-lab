//! An embedded font program, read only for the text of its glyphs and the weight it states of itself.
//!
//! A TrueType/OpenType program (`/FontFile2`, or `/FontFile3` of subtype `OpenType`) gives its `cmap` table
//! reversed (glyph to Unicode) and its `post` table's glyph names -- what a font says about its glyphs is the last
//! word a reader has when the PDF gives no `ToUnicode` `CMap` -- and its `OS/2` weight class; a Type 1 program
//! (`/FontFile`) its built-in encoding and its `/Weight`.

use std::collections::HashMap;

/// Each glyph's text, as the font program states it.
#[derive(Debug, Clone, Default)]
pub struct GlyphText {
    /// Glyph id to text, from the Unicode `cmap` subtables (the first code a glyph is mapped from).
    by_cmap: HashMap<u16, char>,
    /// Glyph id to its `post` table name.
    names: HashMap<u16, String>,
    /// The (3, 0) symbol subtable: a code's glyph (a symbolic font's codes are `0xF000 +` code there).
    symbol: HashMap<u32, u16>,
    /// The (1, 0) Macintosh subtable: a one-byte code's glyph.
    mac: HashMap<u32, u16>,
}

impl GlyphText {
    /// The text of glyph `gid`: by the Unicode `cmap`, else by its `post` name.
    #[must_use]
    pub fn text(&self, gid: u16) -> Option<String> {
        if let Some(&c) = self.by_cmap.get(&gid) {
            return Some(c.to_string());
        }
        self.names
            .get(&gid)
            .and_then(|n| crate::font::glyph_text(n))
    }

    /// The glyph a simple font's one-byte `code` selects through the font's own `cmap` (symbol, then Macintosh).
    #[must_use]
    pub fn glyph_of_code(&self, code: u32) -> Option<u16> {
        self.symbol
            .get(&(0xF000 + code))
            .or_else(|| self.symbol.get(&code))
            .or_else(|| self.mac.get(&code))
            .copied()
    }

    /// Read `data`; empty when it is not a TrueType/OpenType font.
    #[must_use]
    pub fn parse(data: &[u8]) -> Self {
        let mut out = Self::default();
        let Some(tables) = tables(data) else {
            return out;
        };
        if let Some(&(offset, len)) = tables.get(b"cmap") {
            out.read_cmap(data.get(offset..offset + len).unwrap_or(&[]));
        }
        if let Some(&(offset, len)) = tables.get(b"post") {
            out.names = post_names(data.get(offset..offset + len).unwrap_or(&[]));
        }
        out
    }

    fn read_cmap(&mut self, table: &[u8]) {
        let count = usize::from(u16_at(table, 2).unwrap_or(0));
        for i in 0..count {
            let record = 4 + i * 8;
            let (Some(platform), Some(encoding), Some(offset)) = (
                u16_at(table, record),
                u16_at(table, record + 2),
                u32_at(table, record + 4),
            ) else {
                break;
            };
            let Some(sub) = usize::try_from(offset).ok().and_then(|o| table.get(o..)) else {
                continue;
            };
            let map = subtable(sub);
            match (platform, encoding) {
                (0, _) | (3, 1 | 10) => {
                    for (code, gid) in map {
                        if let Some(c) = char::from_u32(code) {
                            self.by_cmap.entry(gid).or_insert(c);
                        }
                    }
                }
                (3, 0) => self.symbol.extend(map),
                (1, 0) => self.mac.extend(map),
                _ => {}
            }
        }
    }
}

/// A Type 1 font program's built-in encoding (`/Encoding ... dup <code> /<name> put ...` in its clear-text
/// part): the glyph name of each code. None when the program uses `StandardEncoding` or states none.
#[must_use]
pub fn type1_encoding(program: &[u8]) -> Option<Vec<Option<String>>> {
    let clear = &program[..crate::parser::find(program, b"eexec").unwrap_or(program.len())];
    let start = crate::parser::find(clear, b"/Encoding")? + b"/Encoding".len();
    let rest = &clear[start..];
    if rest
        .iter()
        .take(40)
        .collect::<Vec<_>>()
        .windows(16)
        .any(|w| w.iter().map(|b| **b).eq(*b"StandardEncoding"))
    {
        return None;
    }
    let end = crate::parser::find(rest, b"readonly def")
        .or_else(|| crate::parser::find(rest, b" def"))
        .unwrap_or(rest.len());
    let mut names = vec![None; 256];
    let mut found = false;
    for op in crate::content::operations(&rest[..end]).0 {
        if op.operator == b"put"
            && let [
                ..,
                crate::object::Object::Integer(code),
                crate::object::Object::Name(name),
            ] = op.operands.as_slice()
            && let Some(slot) = usize::try_from(*code).ok().and_then(|c| names.get_mut(c))
        {
            *slot = Some(String::from_utf8_lossy(name).into_owned());
            found = true;
        }
    }
    found.then_some(names)
}

/// The weight class (100-900) a TrueType/OpenType program states of itself: its `OS/2` table's `usWeightClass`
/// (an old 1-9 scale read as hundreds), else 700 when its `head` table's `macStyle` marks it bold. None when it
/// states neither: a subset program often keeps no `OS/2` table, and an unset bold bit says nothing of a
/// semibold.
#[must_use]
pub fn sfnt_weight(data: &[u8]) -> Option<f64> {
    let tables = tables(data)?;
    let table = |tag: &[u8; 4]| {
        let &(offset, len) = tables.get(tag)?;
        data.get(offset..offset.checked_add(len)?)
    };
    let class = table(b"OS/2").and_then(|os2| u16_at(os2, 4));
    match class {
        Some(c @ 1..=9) => Some(f64::from(c) * 100.0),
        Some(c @ 100..=1000) => Some(f64::from(c)),
        _ => table(b"head")
            .and_then(|head| u16_at(head, 44))
            .filter(|style| style & 1 != 0)
            .map(|_| 700.0),
    }
}

/// The PostScript name a TrueType/OpenType program states of itself: its `name` table's name ID 6, a Windows
/// Unicode record's (UTF-16BE) or else a Macintosh Roman one's (ASCII, the only characters a PostScript name has).
#[must_use]
pub fn sfnt_postscript_name(data: &[u8]) -> Option<String> {
    let &(offset, len) = tables(data)?.get(b"name")?;
    let table = data.get(offset..offset.checked_add(len)?)?;
    let strings = usize::from(u16_at(table, 4)?);
    let mut found = None;
    for i in 0..usize::from(u16_at(table, 2)?) {
        let record = 6 + i * 12;
        let field = |k: usize| u16_at(table, record + 2 * k);
        if field(3)? != 6 {
            continue;
        }
        let start = strings + usize::from(field(5)?);
        let bytes = table.get(start..start + usize::from(field(4)?))?;
        match (field(0)?, field(1)?) {
            (3, 0 | 1) => {
                let units: Vec<u16> = bytes
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|p| u16::from_be_bytes(*p))
                    .collect();
                return String::from_utf16(&units).ok();
            }
            (1, 0) if bytes.is_ascii() => {
                found = Some(String::from_utf8_lossy(bytes).into_owned());
            }
            _ => {}
        }
    }
    found
}

/// The weight class a Type 1 program states of itself: its `/Weight` string (`(Bold)`, `(Medium)`, ...) in its
/// clear-text part, read as a font name's style words are.
#[must_use]
pub fn type1_weight(program: &[u8]) -> Option<f64> {
    let clear = &program[..crate::parser::find(program, b"eexec").unwrap_or(program.len())];
    let rest = &clear[crate::parser::find(clear, b"/Weight")? + b"/Weight".len()..];
    let open = rest.iter().position(|&b| !b.is_ascii_whitespace())?;
    if rest[open] != b'(' {
        return None;
    }
    let close = open + rest[open..].iter().position(|&b| b == b')')?;
    let weight = String::from_utf8_lossy(&rest[open + 1..close]);
    crate::font::style_weight(crate::font::words(&weight))
}

fn u16_at(data: &[u8], at: usize) -> Option<u16> {
    Some(u16::from_be_bytes(data.get(at..at + 2)?.try_into().ok()?))
}

fn u32_at(data: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_be_bytes(data.get(at..at + 4)?.try_into().ok()?))
}

/// The table directory: tag to (offset, length).
fn tables(data: &[u8]) -> Option<HashMap<[u8; 4], (usize, usize)>> {
    let version = u32_at(data, 0)?;
    if !matches!(version, 0x0001_0000 | 0x4F54_544F | 0x7472_7565) {
        return None; // not TrueType (1.0, 'OTTO', 'true')
    }
    let count = usize::from(u16_at(data, 4)?);
    let mut out = HashMap::new();
    for i in 0..count {
        let record = 12 + i * 16;
        let tag: [u8; 4] = data.get(record..record + 4)?.try_into().ok()?;
        let offset = usize::try_from(u32_at(data, record + 8)?).ok()?;
        let len = usize::try_from(u32_at(data, record + 12)?).ok()?;
        out.insert(tag, (offset, len));
    }
    Some(out)
}

/// One `cmap` subtable's (code, glyph) pairs: formats 0, 4, 6 and 12.
fn subtable(sub: &[u8]) -> Vec<(u32, u16)> {
    let mut out = Vec::new();
    match u16_at(sub, 0) {
        Some(0) => {
            for code in 0..256u32 {
                if let Some(&g) = sub.get(6 + code as usize) {
                    out.push((code, u16::from(g)));
                }
            }
        }
        Some(4) => {
            let segs = usize::from(u16_at(sub, 6).unwrap_or(0) / 2);
            let (ends, starts, deltas, offsets) = (14, 16 + segs * 2, 16 + segs * 4, 16 + segs * 6);
            for s in 0..segs {
                let (Some(end), Some(start), Some(delta), Some(range)) = (
                    u16_at(sub, ends + s * 2),
                    u16_at(sub, starts + s * 2),
                    u16_at(sub, deltas + s * 2),
                    u16_at(sub, offsets + s * 2),
                ) else {
                    break;
                };
                for code in start..=end {
                    if code == 0xFFFF {
                        break;
                    }
                    let gid = if range == 0 {
                        code.wrapping_add(delta)
                    } else {
                        let at =
                            offsets + s * 2 + usize::from(range) + usize::from(code - start) * 2;
                        match u16_at(sub, at) {
                            Some(0) | None => 0,
                            Some(g) => g.wrapping_add(delta),
                        }
                    };
                    if gid != 0 {
                        out.push((u32::from(code), gid));
                    }
                }
            }
        }
        Some(6) => {
            let (first, count) = (u16_at(sub, 6).unwrap_or(0), u16_at(sub, 8).unwrap_or(0));
            for i in 0..count {
                if let Some(g) = u16_at(sub, 10 + usize::from(i) * 2) {
                    out.push((u32::from(first) + u32::from(i), g));
                }
            }
        }
        Some(12) => {
            let groups = u32_at(sub, 12).unwrap_or(0);
            for i in 0..usize::try_from(groups.min(0x10000)).unwrap_or(0) {
                let at = 16 + i * 12;
                let (Some(start), Some(end), Some(first)) =
                    (u32_at(sub, at), u32_at(sub, at + 4), u32_at(sub, at + 8))
                else {
                    break;
                };
                for code in start..=end.min(start.saturating_add(0xFFFF)) {
                    if let Ok(g) = u16::try_from(first + (code - start)) {
                        out.push((code, g));
                    }
                }
            }
        }
        _ => {}
    }
    out
}

/// The `post` table's glyph names (format 2: the 258 standard Macintosh names, then the table's own).
fn post_names(table: &[u8]) -> HashMap<u16, String> {
    let mut out = HashMap::new();
    if u32_at(table, 0) != Some(0x0002_0000) {
        return out;
    }
    let count = usize::from(u16_at(table, 32).unwrap_or(0));
    let indices: Vec<u16> = (0..count)
        .filter_map(|i| u16_at(table, 34 + i * 2))
        .collect();
    let mut own = Vec::new();
    let mut at = 34 + count * 2;
    while let Some(&len) = table.get(at) {
        let Some(name) = table.get(at + 1..at + 1 + usize::from(len)) else {
            break;
        };
        own.push(String::from_utf8_lossy(name).into_owned());
        at += 1 + usize::from(len);
    }
    for (gid, &index) in (0u16..).zip(&indices) {
        let name = if index < 258 {
            MAC_NAMES.get(usize::from(index)).map(|s| (*s).to_string())
        } else {
            own.get(usize::from(index) - 258).cloned()
        };
        if let Some(n) = name {
            out.insert(gid, n);
        }
    }
    out
}

/// The 258 standard Macintosh glyph names of `post` format 1 and 2, in order.
const MAC_NAMES: [&str; 258] = [
    ".notdef",
    ".null",
    "nonmarkingreturn",
    "space",
    "exclam",
    "quotedbl",
    "numbersign",
    "dollar",
    "percent",
    "ampersand",
    "quotesingle",
    "parenleft",
    "parenright",
    "asterisk",
    "plus",
    "comma",
    "hyphen",
    "period",
    "slash",
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "colon",
    "semicolon",
    "less",
    "equal",
    "greater",
    "question",
    "at",
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "bracketleft",
    "backslash",
    "bracketright",
    "asciicircum",
    "underscore",
    "grave",
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
    "braceleft",
    "bar",
    "braceright",
    "asciitilde",
    "Adieresis",
    "Aring",
    "Ccedilla",
    "Eacute",
    "Ntilde",
    "Odieresis",
    "Udieresis",
    "aacute",
    "agrave",
    "acircumflex",
    "adieresis",
    "atilde",
    "aring",
    "ccedilla",
    "eacute",
    "egrave",
    "ecircumflex",
    "edieresis",
    "iacute",
    "igrave",
    "icircumflex",
    "idieresis",
    "ntilde",
    "oacute",
    "ograve",
    "ocircumflex",
    "odieresis",
    "otilde",
    "uacute",
    "ugrave",
    "ucircumflex",
    "udieresis",
    "dagger",
    "degree",
    "cent",
    "sterling",
    "section",
    "bullet",
    "paragraph",
    "germandbls",
    "registered",
    "copyright",
    "trademark",
    "acute",
    "dieresis",
    "notequal",
    "AE",
    "Oslash",
    "infinity",
    "plusminus",
    "lessequal",
    "greaterequal",
    "yen",
    "mu",
    "partialdiff",
    "summation",
    "product",
    "pi",
    "integral",
    "ordfeminine",
    "ordmasculine",
    "Omega",
    "ae",
    "oslash",
    "questiondown",
    "exclamdown",
    "logicalnot",
    "radical",
    "florin",
    "approxequal",
    "Delta",
    "guillemotleft",
    "guillemotright",
    "ellipsis",
    "nonbreakingspace",
    "Agrave",
    "Atilde",
    "Otilde",
    "OE",
    "oe",
    "endash",
    "emdash",
    "quotedblleft",
    "quotedblright",
    "quoteleft",
    "quoteright",
    "divide",
    "lozenge",
    "ydieresis",
    "Ydieresis",
    "fraction",
    "currency",
    "guilsinglleft",
    "guilsinglright",
    "fi",
    "fl",
    "daggerdbl",
    "periodcentered",
    "quotesinglbase",
    "quotedblbase",
    "perthousand",
    "Acircumflex",
    "Ecircumflex",
    "Aacute",
    "Edieresis",
    "Egrave",
    "Iacute",
    "Icircumflex",
    "Idieresis",
    "Igrave",
    "Oacute",
    "Ocircumflex",
    "apple",
    "Ograve",
    "Uacute",
    "Ucircumflex",
    "Ugrave",
    "dotlessi",
    "circumflex",
    "tilde",
    "macron",
    "breve",
    "dotaccent",
    "ring",
    "cedilla",
    "hungarumlaut",
    "ogonek",
    "caron",
    "Lslash",
    "lslash",
    "Scaron",
    "scaron",
    "Zcaron",
    "zcaron",
    "brokenbar",
    "Eth",
    "eth",
    "Yacute",
    "yacute",
    "Thorn",
    "thorn",
    "minus",
    "multiply",
    "onesuperior",
    "twosuperior",
    "threesuperior",
    "onehalf",
    "onequarter",
    "threequarters",
    "franc",
    "Gbreve",
    "gbreve",
    "Idotaccent",
    "Scedilla",
    "scedilla",
    "Cacute",
    "cacute",
    "Ccaron",
    "ccaron",
    "dcroat",
];

#[cfg(test)]
mod tests {
    use super::*;

    /// A font of one `cmap` (format 4: 'A'-'C' to glyphs 5-7) and a `post` (format 2: glyph 8 named `alpha`).
    fn font() -> Vec<u8> {
        let mut cmap = vec![0, 0, 0, 1, 0, 3, 0, 1, 0, 0, 0, 12];
        // Format 4, two segments: 0x41-0x43 delta to 5, and the 0xFFFF terminator.
        let seg = |v: [u16; 2]| v.iter().flat_map(|x| x.to_be_bytes()).collect::<Vec<u8>>();
        cmap.extend([0, 4, 0, 32, 0, 0, 0, 4, 0, 4, 0, 1, 0, 0]);
        cmap.extend(seg([0x43, 0xFFFF]));
        cmap.extend([0, 0]);
        cmap.extend(seg([0x41, 0xFFFF]));
        cmap.extend(seg([5u16.wrapping_sub(0x41), 1]));
        cmap.extend(seg([0, 0]));
        let mut post = vec![0, 2, 0, 0];
        post.extend([0; 28]);
        post.extend([0, 9]);
        for i in 0..8u16 {
            post.extend(i.to_be_bytes());
        }
        post.extend(258u16.to_be_bytes());
        post.push(5);
        post.extend(b"alpha");
        let mut out = vec![0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0];
        let cmap_at = 12 + 32;
        let post_at = cmap_at + cmap.len();
        for (tag, at, len) in [
            (b"cmap", cmap_at, cmap.len()),
            (b"post", post_at, post.len()),
        ] {
            out.extend(tag);
            out.extend([0; 4]);
            out.extend(u32::try_from(at).unwrap().to_be_bytes());
            out.extend(u32::try_from(len).unwrap().to_be_bytes());
        }
        out.extend(cmap);
        out.extend(post);
        out
    }

    #[test]
    fn a_glyph_text_comes_from_the_cmap_then_the_post_names() {
        let text = GlyphText::parse(&font());
        assert_eq!(text.text(6).as_deref(), Some("B"));
        assert_eq!(text.text(8).as_deref(), Some("\u{3B1}"));
        assert_eq!(text.text(3).as_deref(), Some(" "));
        assert_eq!(text.text(99), None);
    }

    #[test]
    fn a_type1_program_states_its_own_encoding() {
        let program = b"%!PS-AdobeFont-1.0: CMR12\n/Encoding 256 array\n0 1 255 {1 index exch /.notdef put} for\n\
            dup 65 /A put\ndup 11 /ff put\nreadonly def\ncurrentfile eexec \x80\x81";
        let names = type1_encoding(program).unwrap();
        assert_eq!(
            (
                names[65].as_deref(),
                names[11].as_deref(),
                names[66].as_deref()
            ),
            (Some("A"), Some("ff"), None)
        );
        assert_eq!(type1_encoding(b"/Encoding StandardEncoding def"), None);
    }

    #[test]
    fn data_that_is_no_font_reads_as_no_text() {
        assert_eq!(GlyphText::parse(b"%!PS-AdobeFont").text(1), None);
    }

    /// A program of the tables given, each `(tag, data)`.
    fn sfnt(tables: &[(&[u8; 4], Vec<u8>)]) -> Vec<u8> {
        let count = u16::try_from(tables.len()).unwrap();
        let mut out = vec![0, 1, 0, 0];
        out.extend(count.to_be_bytes());
        out.extend([0; 6]);
        let mut at = 12 + 16 * tables.len();
        for (tag, data) in tables {
            out.extend(*tag);
            out.extend([0; 4]);
            out.extend(u32::try_from(at).unwrap().to_be_bytes());
            out.extend(u32::try_from(data.len()).unwrap().to_be_bytes());
            at += data.len();
        }
        for (_, data) in tables {
            out.extend(data);
        }
        out
    }

    #[test]
    fn a_truetype_program_states_its_weight_class_else_its_bold_bit() {
        let os2 = |class: u16| [&[0, 4, 0, 0][..], &class.to_be_bytes()].concat();
        let head = |style: u16| [&[0; 44][..], &style.to_be_bytes()].concat();
        assert_eq!(sfnt_weight(&sfnt(&[(b"OS/2", os2(600))])), Some(600.0));
        assert_eq!(sfnt_weight(&sfnt(&[(b"OS/2", os2(7))])), Some(700.0));
        assert_eq!(sfnt_weight(&sfnt(&[(b"head", head(1))])), Some(700.0));
        assert_eq!(sfnt_weight(&sfnt(&[(b"head", head(2))])), None);
        assert_eq!(
            sfnt_weight(&sfnt(&[(b"OS/2", os2(400)), (b"head", head(1))])),
            Some(400.0)
        );
        assert_eq!(sfnt_weight(b"%!PS-AdobeFont"), None);
    }

    #[test]
    fn a_type1_program_states_its_weight_in_its_clear_text() {
        let program = b"%!PS-AdobeFont-1.0: CMBX12\n/FontInfo 9 dict dup begin\n/Weight (Bold) readonly def\n\
            end readonly def\ncurrentfile eexec \x80\x81";
        assert_eq!(type1_weight(program), Some(700.0));
        assert_eq!(type1_weight(b"/Weight (Medium) readonly def"), Some(500.0));
        assert_eq!(
            type1_weight(b"/FontName /CMR12 def eexec /Weight (Bold)"),
            None
        );
    }
}
