//! Fonts (ISO 32000-2 section 9.5-9.10), read for what text needs: how a string splits into codes, each code's
//! width and Unicode text, the font's ascent and descent, and whether it is bold ([`FontInfo::bold`]); and for what
//! drawing it needs, each code's glyph shape ([`Font::shape`], [`crate::shape`]).

use crate::base14::{FONTS, Metrics};
use crate::cid2unicode::Collection;
use crate::cmap::{CMap, legacy_encoding};
use crate::error::Result;
use crate::file::File;
use crate::fontfile::{self, GlyphText};
use crate::geometry::Matrix;
use crate::object::{Dict, Object, Stream};
use crate::shape::{self, Shapes, Source};
use crate::symbol_tables::{CORPORATE_USE, WEBDINGS, WINGDINGS, WINGDINGS_2, WINGDINGS_3};
use std::rc::Rc;

/// What identifies a font to a reader: its name, the descriptor's facts and what its program says of itself.
#[derive(Debug, Clone, PartialEq)]
pub struct FontInfo {
    /// `/BaseFont` (a subset tag such as `ABCDEF+` kept), or `/Name` for a Type 3 font.
    pub name: String,
    /// `/Subtype`: `Type1`, `TrueType`, `Type0`, `Type3`, `MMType1`.
    pub subtype: String,
    /// The descriptor's `/Flags` (bit 19 `ForceBold`, bit 7 `Italic`, ...); 0 when absent.
    pub flags: i64,
    /// The descriptor's `/FontWeight` (100-900), when stated.
    pub descriptor_weight: Option<f64>,
    /// The weight class (100-900) the embedded program states of itself, when it states one this reader reads: a
    /// TrueType/OpenType program's `OS/2` `usWeightClass` (or its `head` `macStyle` bold bit), a Type 1 program's
    /// `/Weight`, a bare CFF program's (`/FontFile3` `Type1C`, `CIDFontType0C`) Top DICT `Weight`, else its
    /// `FullName`'s or font name's style part.
    pub program_weight: Option<f64>,
    /// The descriptor's `/ItalicAngle` (degrees), when stated.
    pub italic_angle: Option<f64>,
    /// Whether the font program is embedded.
    pub embedded: bool,
}

/// The descriptor's `ForceBold` flag (bit 19): set only for bold glyphs, to keep their stems thick at small sizes.
const FORCE_BOLD: i64 = 1 << 18;
/// A weight class this heavy or heavier is bold (600: semibold).
const BOLD_WEIGHT: f64 = 600.0;

impl FontInfo {
    /// Its name without a subset tag (ISO 32000-2 9.6.4: six capital letters and `+`, as `ABCDEF+Helvetica`).
    #[must_use]
    pub fn name_without_subset(&self) -> &str {
        without_subset(&self.name)
    }

    /// The weight class its name states: that of the last weight word (`Bold`, `Semibold`, `Black`, `Light`, ...)
    /// among its words after the first -- its style part, by the `Family-Style` convention of PostScript names
    /// (`TimesNewRomanPS-BoldMT`) and the `Family,Style` one ISO 32000-2 9.6.2.2 gives a TrueType `/BaseFont`
    /// (`Arial,Bold`). The first word is the family's: a family named `Black...` is not black.
    #[must_use]
    pub fn name_weight(&self) -> Option<f64> {
        name_style_weight(&self.name)
    }

    /// Whether it is bold: its descriptor sets `ForceBold`, or a statement of its weight -- the descriptor's
    /// `/FontWeight`, its program's own weight class, its name's style part -- is 600 (semibold) or heavier.
    ///
    /// Any one statement suffices: none states a weight lighter than the glyphs shown (a regular program shown
    /// under a `,Bold` name is drawn emboldened), and a missing flag or word states nothing. The name is read
    /// because a font often states its weight nowhere else -- a TrueType font the file does not embed has its
    /// style in its name alone -- and only its style part is read, so a family name holding a weight word states
    /// nothing.
    #[must_use]
    pub fn bold(&self) -> bool {
        self.flags & FORCE_BOLD != 0
            || [
                self.descriptor_weight,
                self.program_weight,
                self.name_weight(),
            ]
            .into_iter()
            .flatten()
            .any(|w| w >= BOLD_WEIGHT)
    }
}

impl FontInfo {
    /// The facts of the font named `name`, of `subtype`, its descriptor stating them (its values resolved in
    /// `file`) and its embedded program stating `program_weight`.
    ///
    /// Refuses a reference in the descriptor that does not resolve.
    fn new(
        file: &File,
        name: String,
        subtype: String,
        descriptor: Option<&Dict>,
        program_weight: Option<f64>,
    ) -> Result<Self> {
        Ok(Self {
            name,
            subtype,
            flags: file.flags(descriptor)?,
            descriptor_weight: file.number(descriptor, b"FontWeight")?,
            program_weight,
            italic_angle: file.number(descriptor, b"ItalicAngle")?,
            embedded: descriptor.is_some_and(|d| {
                [&b"FontFile"[..], b"FontFile2", b"FontFile3"]
                    .iter()
                    .any(|k| d.get(k).is_some())
            }),
        })
    }
}

/// `name` without its subset tag (six capital letters and `+`), if it has one.
fn without_subset(name: &str) -> &str {
    match name.split_at_checked(7) {
        Some((tag, rest))
            if tag.ends_with('+') && tag.bytes().take(6).all(|b| b.is_ascii_uppercase()) =>
        {
            rest
        }
        _ => name,
    }
}

/// The words of a font name or style: split at `,`, `-`, `_` and spaces, before a capital following a small
/// letter (`BoldItalic`), and between letters and digits (`LMRoman10`).
pub(crate) fn words(name: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0;
    let mut previous: Option<char> = None;
    for (i, c) in name.char_indices() {
        if matches!(c, ',' | '-' | '_' | ' ') {
            out.push(&name[start..i]);
            start = i + c.len_utf8();
            previous = None;
            continue;
        }
        if let Some(p) = previous
            && ((p.is_lowercase() && c.is_uppercase()) || p.is_ascii_digit() != c.is_ascii_digit())
        {
            out.push(&name[start..i]);
            start = i;
        }
        previous = Some(c);
    }
    out.push(&name[start..]);
    out.retain(|w| !w.is_empty());
    out
}

/// The weight class the style part of the font name `name` states -- its words after the first, a subset tag
/// dropped -- as [`FontInfo::name_weight`] reads it.
pub(crate) fn name_style_weight(name: &str) -> Option<f64> {
    style_weight(words(without_subset(name)).into_iter().skip(1))
}

/// Weight words, lower-cased, and their weight classes (100-900), the abbreviations of PostScript names (`Bd`,
/// `Blk`) among them. `Semi`, `Demi`, `Extra` and `Ultra` join the word after them (`Semi Bold`, `DemiLight`);
/// `Demi` alone is demibold.
const WEIGHTS: [(&str, f64); 26] = [
    ("thin", 100.0),
    ("hairline", 100.0),
    ("extralight", 200.0),
    ("ultralight", 200.0),
    ("light", 300.0),
    ("lt", 300.0),
    ("semilight", 350.0),
    ("demilight", 350.0),
    ("book", 400.0),
    ("normal", 400.0),
    ("regular", 400.0),
    ("roman", 400.0),
    ("medium", 500.0),
    ("md", 500.0),
    ("semibold", 600.0),
    ("demibold", 600.0),
    ("demi", 600.0),
    ("bold", 700.0),
    ("bd", 700.0),
    ("extrabold", 800.0),
    ("ultrabold", 800.0),
    ("heavy", 800.0),
    ("hv", 800.0),
    ("black", 900.0),
    ("blk", 900.0),
    ("extrablack", 950.0),
];

/// The weight class `words` state: their last weight word's; None when they hold none.
pub(crate) fn style_weight<'a>(words: impl IntoIterator<Item = &'a str>) -> Option<f64> {
    let class = |w: &str| {
        WEIGHTS
            .iter()
            .find(|(name, _)| name.eq_ignore_ascii_case(w))
            .map(|&(_, class)| class)
    };
    let mut words = words.into_iter().map(str::to_ascii_lowercase).peekable();
    let mut found = None;
    while let Some(word) = words.next() {
        let joined = matches!(word.as_str(), "semi" | "demi" | "extra" | "ultra")
            .then(|| {
                words
                    .peek()
                    .and_then(|next| class(&format!("{word}{next}")))
            })
            .flatten();
        if joined.is_some() {
            words.next();
        }
        found = joined.or_else(|| class(&word)).or(found);
    }
    found
}

/// A value for each one-byte code of a simple font, None for a code it gives none.
type ByCode = Vec<Option<String>>;

/// How codes become glyphs.
#[derive(Debug, Clone)]
pub(crate) enum Kind {
    /// One byte per code; widths by code; Unicode by the encoding; glyphs by its names.
    Simple {
        first: i64,
        widths: Vec<f64>,
        missing: f64,
        encoding: Vec<Option<String>>,
        /// Each code's glyph name, as its encoding states it; None where the program's own encoding selects.
        names: Vec<Option<String>>,
        base14: Option<&'static Metrics>,
    },
    /// Codes by a `CMap`, widths by `CID`; vertical displacements and position vectors by `CID` when the `CMap`
    /// writes vertically.
    Composite {
        cmap: CMap,
        widths: CidMetric,
        vertical: Option<Box<Vertical>>,
        code_text: CodeText,
        /// Adobe's text of the `CID`s, of the collection a predefined `CMap` is of, or for Identity and an embedded
        /// `CMap` the one the `CIDFont`'s `/CIDSystemInfo` names, when Adobe's tables cover it. The ordering decides,
        /// not the font's kind: a `CIDFontType2` declaring `Adobe-Identity` has `CID`s that are only glyph ids, and
        /// none; one declaring `Adobe-Japan1` has that collection's `CID`s (its `/CIDToGIDMap` turns them into
        /// glyphs).
        collection: Option<&'static Collection>,
    },
}

/// One metric of a composite font's glyphs by `CID`, in glyph space (9.7.4.3): `/W`'s widths and `/DW`, or a
/// vertical font's displacements w1y from `/W2` and `/DW2`.
#[derive(Debug, Clone)]
pub(crate) struct CidMetric {
    /// (first `CID`, last `CID`, value) runs, in the order stated: the first run holding a `CID` gives its value.
    runs: Vec<(u32, u32, f64)>,
    /// The value of a `CID` no run holds.
    default: f64,
}

impl CidMetric {
    fn of(&self, cid: u32) -> f64 {
        self.find(cid).unwrap_or(self.default)
    }

    fn find(&self, cid: u32) -> Option<f64> {
        self.runs
            .iter()
            .find(|(lo, hi, _)| (*lo..=*hi).contains(&cid))
            .map(|run| run.2)
    }
}

/// A vertical font's metrics by `CID` (9.7.4.3), in glyph space: the displacement w1y, and the position vector
/// (vx, vy) from the horizontal origin to the vertical one, which places the glyph's outline -- `/W2`'s, else vx
/// half the glyph's width and vy `/DW2`'s first number.
#[derive(Debug, Clone)]
pub(crate) struct Vertical {
    displacement: CidMetric,
    across: CidMetric,
    down: CidMetric,
}

/// What a composite font's codes are, as text, when no `ToUnicode` `CMap` says.
#[derive(Debug, Clone, Copy)]
pub(crate) enum CodeText {
    /// Nothing: the codes select glyphs only.
    Unknown,
    /// UTF-16 code units (a predefined `Uni...-UCS2-` or `-UTF16-` `CMap`).
    Utf16,
    /// Bytes of a legacy encoding (a predefined `GBK-EUC-H`, `B5pc-H`, `90ms-RKSJ-H`, ... `CMap`).
    Legacy(&'static encoding_rs::Encoding),
}

/// A font, read.
#[derive(Debug, Clone)]
pub struct Font {
    /// Its identity.
    pub info: Rc<FontInfo>,
    pub(crate) kind: Kind,
    pub(crate) to_unicode: Option<CMap>,
    /// The embedded TrueType program's glyph text, the last word on a glyph's text.
    glyph_text: Option<GlyphText>,
    /// A `CIDFontType2`'s CID-to-glyph map (None: the identity).
    pub(crate) cid_to_gid: Option<Vec<u16>>,
    /// The symbol font it is, by its name or its program's: its private-use text is read as the symbol shown.
    pub(crate) symbols: Option<Symbols>,
    /// Its glyphs' shapes.
    pub(crate) shapes: Shapes,
    /// From glyph space to text space (`[0.001 0 0 0.001 0 0]` but for Type 3 fonts).
    pub matrix: Matrix,
    /// The ascent in glyph space.
    pub ascent: f64,
    /// The descent in glyph space (negative).
    pub descent: f64,
    /// What of it could not be read, in words (a program too malformed to state its weight); empty when all was.
    pub warnings: Vec<String>,
}

/// One code of a string: its value, Unicode text and advances in text space (before the font size).
#[derive(Debug, Clone, PartialEq)]
pub struct Code {
    /// The code.
    pub code: u32,
    /// How many bytes it took.
    pub bytes: usize,
    /// Its text, when the font says what it is.
    pub text: Option<String>,
    /// Its horizontal advance w0 in text space units of one font size.
    pub width: f64,
    /// Its vertical advance w1y in text space units of one font size (negative: down the column), when its font
    /// writes vertically.
    pub vertical_advance: Option<f64>,
    /// Its position vector (vx, vy) in text space units of one font size, when its font writes vertically: where
    /// its horizontal origin, which its outline is drawn from, lies before its vertical origin.
    pub position: Option<(f64, f64)>,
}

const GLYPH_SPACE: Matrix = Matrix([0.001, 0.0, 0.0, 0.001, 0.0, 0.0]);

/// A font is equal to itself alone: a font dictionary read once is one font, shared through the file's cache.
impl PartialEq for Font {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self, other)
    }
}

impl Font {
    /// Whether the font writes vertically: its `CMap`'s writing mode is 1 (a predefined vertical `CMap`, `Identity-V`,
    /// an embedded one with `/WMode 1`), so its glyphs advance down a column (ISO 32000-2 9.7.4.3).
    #[must_use]
    pub fn vertical(&self) -> bool {
        matches!(
            self.kind,
            Kind::Composite {
                vertical: Some(_),
                ..
            }
        )
    }

    /// The glyph name a simple font's encoding gives `code`; None for a composite font, or where the program's own
    /// encoding selects its glyph.
    pub(crate) fn glyph_name(&self, code: u32) -> Option<String> {
        match &self.kind {
            Kind::Simple { names, .. } => names.get(usize::try_from(code).ok()?)?.clone(),
            Kind::Composite { .. } => None,
        }
    }

    /// Whether its descriptor flags it symbolic (Table 121 bit 3): its characters are outside the standard Latin
    /// set, and a TrueType program's codes select its glyphs directly.
    pub(crate) fn symbolic(&self) -> bool {
        self.info.flags & 4 != 0
    }

    /// The width the font states for `code`, in text space units of one font size: a simple font's `/Widths`
    /// entry, None when it states none; a composite font's `/W` entry for the code's `CID`, else its `/DW`.
    pub(crate) fn stated_width(&self, code: u32) -> Option<f64> {
        let width = match &self.kind {
            Kind::Simple { first, widths, .. } => {
                *widths.get(usize::try_from(i64::from(code) - first).ok()?)?
            }
            Kind::Composite { cmap, widths, .. } => widths.of(cmap.cid(code)),
        };
        Some(width * self.matrix.0[0])
    }

    /// Split `bytes` into codes, each with its text and advances.
    #[must_use]
    pub fn codes(&self, bytes: &[u8]) -> Vec<Code> {
        let units = self.matrix.0[0];
        match &self.kind {
            Kind::Simple {
                first,
                widths,
                missing,
                encoding,
                names,
                base14,
            } => bytes
                .iter()
                .map(|&b| {
                    let code = u32::from(b);
                    let text = self
                        .to_unicode
                        .as_ref()
                        .and_then(|c| c.text(code))
                        .map(str::to_string)
                        .or_else(|| encoding[usize::from(b)].clone())
                        .or_else(|| {
                            let program = self.glyph_text.as_ref()?;
                            program.text(program.glyph_of_code(code)?)
                        });
                    let text = match self.symbols {
                        Some(symbols) => symbols.text(text, Some(b)),
                        None => text,
                    };
                    let stated = usize::try_from(i64::from(b) - first)
                        .ok()
                        .and_then(|i| widths.get(i))
                        .copied();
                    // A standard font's width is its glyph's: the one the code's glyph name names (in the Adobe Glyph
                    // List, or ITC Zapf Dingbats' list), else its text's.
                    let glyph = names[usize::from(b)].as_deref().and_then(|n| {
                        glyph_text(n)
                            .or_else(|| skrifa::raw::ps::agl::name_to_char(n).map(String::from))
                    });
                    let width = stated
                        .or_else(|| {
                            base14.and_then(|m| {
                                standard_width(m, glyph.as_deref().or(text.as_deref()))
                            })
                        })
                        .unwrap_or(*missing);
                    Code {
                        code,
                        bytes: 1,
                        text,
                        width: width * units,
                        vertical_advance: None,
                        position: None,
                    }
                })
                .collect(),
            Kind::Composite {
                cmap,
                widths,
                vertical,
                code_text,
                collection,
            } => cmap
                .codes(bytes)
                .into_iter()
                .map(|(code, len)| {
                    let cid = cmap.cid(code);
                    // The text: `ToUnicode`'s, else the code's when codes are Unicode, else the embedded program's
                    // glyph's (the very glyph drawn), else the collection's for the `CID`, else the code's in its
                    // legacy encoding, else none.
                    let text = self
                        .to_unicode
                        .as_ref()
                        .and_then(|c| c.text(code))
                        .map(str::to_string)
                        .or_else(|| unicode_text(*code_text, code, len))
                        .or_else(|| {
                            let gid = match &self.cid_to_gid {
                                Some(map) => *map.get(usize::try_from(cid).ok()?)?,
                                None => u16::try_from(cid).ok()?,
                            };
                            self.glyph_text.as_ref()?.text(gid)
                        })
                        .or_else(|| collection.and_then(|c| c.text(cid)))
                        .or_else(|| legacy_text(*code_text, code, len));
                    let text = match self.symbols {
                        Some(symbols) => symbols.text(text, None),
                        None => text,
                    };
                    let width = widths.of(cid);
                    Code {
                        code,
                        bytes: len,
                        text,
                        width: width * units,
                        vertical_advance: vertical.as_ref().map(|v| v.displacement.of(cid) * units),
                        position: vertical.as_ref().map(|v| {
                            let across = v.across.find(cid).unwrap_or(width / 2.0);
                            (across * units, v.down.of(cid) * units)
                        }),
                    }
                })
                .collect(),
        }
    }
}

/// A code as text when the codes are Unicode: one UTF-16 code unit, or two (a surrogate pair, four bytes).
fn unicode_text(kind: CodeText, code: u32, len: usize) -> Option<String> {
    if !matches!(kind, CodeText::Utf16) {
        return None;
    }
    let units = if len == 4 {
        vec![
            u16::try_from(code >> 16).ok()?,
            u16::try_from(code & 0xFFFF).ok()?,
        ]
    } else {
        vec![u16::try_from(code).ok()?]
    };
    char::decode_utf16(units)
        .collect::<std::result::Result<String, _>>()
        .ok()
}

/// A code as text when its bytes are a legacy encoding's.
fn legacy_text(kind: CodeText, code: u32, len: usize) -> Option<String> {
    let CodeText::Legacy(encoding) = kind else {
        return None;
    };
    let bytes = &code.to_be_bytes()[4 - len.min(4)..];
    let (text, _, failed) = encoding.decode(bytes);
    (!failed).then(|| text.into_owned())
}

/// A standard font's width of the character `text`.
fn standard_width(metrics: &Metrics, text: Option<&str>) -> Option<f64> {
    let c = u32::from(text?.chars().next()?);
    metrics
        .widths
        .binary_search_by_key(&c, |&(u, _)| u)
        .ok()
        .map(|i| f64::from(metrics.widths[i].1))
}

/// The standard font a `/BaseFont` names (or is a common alias of), if any.
fn base14(name: &str) -> Option<&'static Metrics> {
    let canonical = match without_subset(name).replace(' ', "").as_str() {
        "Arial" | "ArialMT" | "Helvetica" => "Helvetica",
        "Arial,Bold" | "Arial-BoldMT" | "Helvetica-Bold" | "Helvetica,Bold" => "Helvetica-Bold",
        "Arial,Italic" | "Arial-ItalicMT" | "Helvetica-Oblique" | "Helvetica,Italic" => {
            "Helvetica-Oblique"
        }
        "Arial,BoldItalic" | "Arial-BoldItalicMT" | "Helvetica-BoldOblique" => {
            "Helvetica-BoldOblique"
        }
        "TimesNewRoman" | "TimesNewRomanPSMT" | "Times-Roman" | "Times" => "Times-Roman",
        "TimesNewRoman,Bold" | "TimesNewRomanPS-BoldMT" | "Times-Bold" => "Times-Bold",
        "TimesNewRoman,Italic" | "TimesNewRomanPS-ItalicMT" | "Times-Italic" => "Times-Italic",
        "TimesNewRoman,BoldItalic" | "TimesNewRomanPS-BoldItalicMT" | "Times-BoldItalic" => {
            "Times-BoldItalic"
        }
        "CourierNew" | "CourierNewPSMT" | "Courier" => "Courier",
        "CourierNew,Bold" | "CourierNewPS-BoldMT" | "Courier-Bold" => "Courier-Bold",
        "CourierNew,Italic" | "CourierNewPS-ItalicMT" | "Courier-Oblique" => "Courier-Oblique",
        "CourierNew,BoldItalic" | "CourierNewPS-BoldItalicMT" | "Courier-BoldOblique" => {
            "Courier-BoldOblique"
        }
        "Symbol" => "Symbol",
        "ZapfDingbats" => "ZapfDingbats",
        _ => return None,
    };
    FONTS.iter().find(|m| m.name == canonical)
}

/// A symbol font: one whose codes show symbols by an encoding of its own, fixed by the font, which its text often
/// states only as private-use code points -- U+F000 plus the code, the (3,0) Microsoft Symbol `cmap`'s convention a
/// symbolic TrueType font follows (ISO 32000-2 9.6.5.4), written into `ToUnicode` `CMap`s as found.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Symbols {
    /// Symbol: Annex D.5's encoding.
    Symbol,
    /// ITC Zapf Dingbats: Annex D.6's encoding.
    ZapfDingbats,
    /// Microsoft's Wingdings, its codes' characters as Unicode 7.0 encoded them (ISO/IEC JTC1/SC2/WG2 N4384), as
    /// for the three below.
    Wingdings,
    /// Microsoft's Wingdings 2.
    Wingdings2,
    /// Microsoft's Wingdings 3.
    Wingdings3,
    /// Microsoft's Webdings.
    Webdings,
}

impl Symbols {
    /// The symbol font a `/BaseFont` names: by its family, the name less a subset tag, a style after `,` or `-`
    /// (`Symbol,Bold`, `Wingdings-Regular`) and spaces (`Wingdings 2`).
    fn named(name: &str) -> Option<Self> {
        let family: String = without_subset(name)
            .split([',', '-'])
            .next()
            .unwrap_or_default()
            .chars()
            .filter(|c| *c != ' ')
            .collect();
        Some(match family.as_str() {
            "Symbol" | "SymbolMT" | "SymbolStd" => Self::Symbol,
            "ZapfDingbats" | "ZapfDingbatsITC" | "ZapfDingbatsStd" => Self::ZapfDingbats,
            "Wingdings" => Self::Wingdings,
            "Wingdings2" => Self::Wingdings2,
            "Wingdings3" => Self::Wingdings3,
            "Webdings" => Self::Webdings,
            _ => return None,
        })
    }

    /// The character code `code` shows: Symbol's and Zapf Dingbats' by their glyph names' text, the Microsoft fonts'
    /// by their tables.
    pub(crate) fn character(self, code: u8) -> Option<char> {
        let table: &[(u8, char)] = match self {
            Self::Symbol => return pdf_encoding::SYMBOL.get(code).map(glyph_of_table),
            Self::ZapfDingbats => return pdf_encoding::ZDINGBAT.get(code).map(glyph_of_table),
            Self::Wingdings => &WINGDINGS,
            Self::Wingdings2 => &WINGDINGS_2,
            Self::Wingdings3 => &WINGDINGS_3,
            Self::Webdings => &WEBDINGS,
        };
        table
            .binary_search_by_key(&code, |&(c, _)| c)
            .ok()
            .map(|i| table[i].1)
    }

    /// The text of a code given `stated`, what the font says of it (`ToUnicode`, encoding, program), and the code
    /// itself for a simple font: a stated real character as stated; a stated private-use U+F000 + c as the
    /// character c shows (left as stated where the font shows none: the Windows logo); in Symbol and Zapf Dingbats a
    /// stated code point of Adobe's corporate use as the character Unicode has since given its glyph; nothing
    /// stated, a simple font's code as the character it shows (the font's (3,0) `cmap` selects its glyph by it).
    fn text(self, stated: Option<String>, code: Option<u8>) -> Option<String> {
        let Some(stated) = stated else {
            return code.and_then(|c| self.character(c)).map(String::from);
        };
        let mut chars = stated.chars();
        let (Some(c), None) = (chars.next(), chars.next()) else {
            return Some(stated);
        };
        let shown = match u8::try_from(u32::from(c).wrapping_sub(0xF000)) {
            Ok(code) => self.character(code),
            Err(_) if matches!(self, Self::Symbol | Self::ZapfDingbats) => corporate_use(c),
            Err(_) => None,
        };
        Some(shown.map_or(stated, String::from))
    }
}

/// The character Unicode has given the glyph Adobe's corporate-use code point `c` stands for, when it has given one.
fn corporate_use(c: char) -> Option<char> {
    CORPORATE_USE
        .binary_search_by_key(&c, |&(u, _)| u)
        .ok()
        .map(|i| CORPORATE_USE[i].1)
}

impl File {
    /// The font `dict` describes.
    ///
    /// # Errors
    ///
    /// Refuses a reference in it that does not resolve.
    pub fn font(&self, dict: &Dict) -> Result<Font> {
        let name_of = |key: &[u8]| -> Result<Option<String>> {
            Ok(self
                .get(dict, key)?
                .and_then(|o| o.as_name().map(|n| String::from_utf8_lossy(n).into_owned())))
        };
        let subtype = name_of(b"Subtype")?.unwrap_or_default();
        let base_name = match name_of(b"BaseFont")? {
            Some(name) => name,
            None => name_of(b"Name")?.unwrap_or_default(),
        };
        let to_unicode = match self.get(dict, b"ToUnicode")? {
            Some(Object::Stream(s)) => self
                .decode(&s)
                .ok()
                .and_then(|d| d.into_bytes().ok())
                .map(|d| CMap::parse(&d)),
            _ => None,
        };
        // A composite font's metrics and descriptor are its descendant CID font's.
        let cid_font = if subtype == "Type0" {
            Some(self.cid_font(dict)?)
        } else {
            None
        };
        let descriptor = self.descriptor(cid_font.as_ref().unwrap_or(dict))?;
        let program = self.program(descriptor.as_ref())?;
        // The symbol font it is: by its name, else -- a producer's generic name, `CIDFont+F7` -- by its program's own.
        let symbols = Symbols::named(&base_name).or_else(|| {
            program
                .as_ref()
                .and_then(Program::postscript_name)
                .and_then(|name| Symbols::named(&name))
        });
        let kind = match &cid_font {
            Some(cid_font) => self.composite(dict, cid_font)?,
            None => self.simple(
                dict,
                &subtype,
                &base_name,
                symbols,
                descriptor.as_ref(),
                program.as_ref(),
            )?,
        };
        let matrix = if subtype == "Type3" {
            match self.get(dict, b"FontMatrix")? {
                Some(Object::Array(m)) => Matrix::from_numbers(
                    &m.iter().filter_map(Object::as_number).collect::<Vec<_>>(),
                )
                .unwrap_or(GLYPH_SPACE),
                _ => GLYPH_SPACE,
            }
        } else {
            GLYPH_SPACE
        };
        let (ascent, descent) = self.extent(
            dict,
            descriptor.as_ref(),
            &subtype,
            base14(&base_name),
            matrix,
        )?;
        let cid_to_gid = match &cid_font {
            Some(cid_font) => self.cid_to_gid(cid_font)?,
            None => None,
        };
        let (program_weight, warnings) = program_weight(program.as_ref());
        let info = FontInfo::new(
            self,
            base_name,
            subtype,
            descriptor.as_ref(),
            program_weight,
        )?;
        let source = self.shape_source(dict, &info, &kind, symbols, program.as_ref())?;
        Ok(Font {
            kind,
            to_unicode,
            glyph_text: match program {
                Some(Program::Sfnt(data)) => Some(GlyphText::parse(&data)),
                _ => None,
            },
            cid_to_gid,
            symbols,
            matrix,
            ascent,
            descent,
            warnings,
            shapes: Shapes::new(source),
            info: Rc::new(info),
        })
    }

    /// The ascent and descent, in glyph space (a Type 3 font's its own, through its `matrix`), of the font `dict`
    /// describes, of `subtype`, its `descriptor` and the `standard` font it names: the descriptor's, else the
    /// standard font's, else the font box's, else 0.8 and -0.2 of the size (the proportions of a Latin text face).
    fn extent(
        &self,
        dict: &Dict,
        descriptor: Option<&Dict>,
        subtype: &str,
        standard: Option<&Metrics>,
        matrix: Matrix,
    ) -> Result<(f64, f64)> {
        let bbox = match descriptor {
            Some(d) => self.rect(d, b"FontBBox")?,
            None if subtype == "Type3" => self.rect(dict, b"FontBBox")?,
            None => None,
        };
        let unit = 0.001 / matrix.0[3].abs().max(f64::MIN_POSITIVE);
        let ascent = self
            .number(descriptor, b"Ascent")?
            .filter(|a| *a > 0.0)
            .or(standard.map(|m| m.ascent))
            .or(bbox.map(|b| b.y1))
            .unwrap_or(800.0 * unit);
        let descent = self
            .number(descriptor, b"Descent")?
            .filter(|d| *d < 0.0)
            .or(standard.map(|m| m.descent))
            .or(bbox.map(|b| b.y0))
            .unwrap_or(-200.0 * unit);
        Ok((ascent, descent))
    }

    /// Where the glyph shapes of the font `dict` describes, of identity `info` and `kind`, come from: a Type 3 font's
    /// procedures, the program it embeds, else its substitute -- for a simple font, by its name or descriptor
    /// ([`shape::substitute`]) -- else its fallbacks -- a CJK composite font's, a Microsoft symbol font's
    /// ([`shape::fallback`]) -- else nowhere, why named.
    fn shape_source(
        &self,
        dict: &Dict,
        info: &FontInfo,
        kind: &Kind,
        symbols: Option<Symbols>,
        program: Option<&Program>,
    ) -> Result<Source> {
        if info.subtype == "Type3" {
            let procedures = match self.get(dict, b"CharProcs")? {
                Some(Object::Dict(d)) => d,
                _ => Dict::new(),
            };
            let resources = match self.get(dict, b"Resources")? {
                Some(Object::Dict(d)) => Some(d),
                _ => None,
            };
            return Ok(Source::Procedures {
                procedures,
                resources,
            });
        }
        if let Some(program) = program {
            return Ok(Source::Embedded(Rc::from(program.data())));
        }
        if info.embedded {
            return Ok(Source::Unavailable(
                format!("font {}: its embedded program does not decode", info.name).into(),
            ));
        }
        let standard = base14(&info.name).map(|m| m.name);
        let italic = shape::italic(info.flags, info.italic_angle, info.name_without_subset());
        let composite = match kind {
            Kind::Composite { collection, .. } => Some(collection.map(|c| c.ordering)),
            Kind::Simple { .. } => None,
        };
        let substitute = shape::substitute(standard, symbols, info.flags, info.bold(), italic);
        let fallback = shape::fallback(composite.flatten(), symbols, info.flags);
        Ok(match (composite, substitute, fallback) {
            (None, Some((substitute, stretched)), _) => Source::Substitute {
                substitute,
                stretched,
            },
            (_, _, Some(chain)) => Source::Fallback {
                fallbacks: self.fallbacks.clone(),
                chain,
                stretched: composite.is_none()
                    || self.fallback_widths.get() == shape::FallbackWidths::Stated,
            },
            _ => Source::Unavailable(
                format!("font {} is not embedded and has no substitute", info.name).into(),
            ),
        })
    }

    /// The program `descriptor` embeds, decoded; None when it embeds none, or one that does not decode.
    fn program(&self, descriptor: Option<&Dict>) -> Result<Option<Program>> {
        let Some(descriptor) = descriptor else {
            return Ok(None);
        };
        let decoded = |stream: &Stream| self.decode(stream).ok().and_then(|d| d.into_bytes().ok());
        let subtype = |stream: &Stream| -> Result<Option<Vec<u8>>> {
            Ok(self
                .get(&stream.dict, b"Subtype")?
                .as_ref()
                .and_then(Object::as_name)
                .map(<[u8]>::to_vec))
        };
        let open_type = |stream: &Stream| {
            Ok::<_, crate::error::Error>(subtype(stream)?.as_deref() == Some(b"OpenType"))
        };
        let bare_cff = |stream: &Stream| {
            Ok::<_, crate::error::Error>(matches!(
                subtype(stream)?.as_deref(),
                Some(b"Type1C" | b"CIDFontType0C")
            ))
        };
        Ok(
            match (
                self.get(descriptor, b"FontFile")?,
                self.get(descriptor, b"FontFile2")?,
                self.get(descriptor, b"FontFile3")?,
            ) {
                (_, Some(Object::Stream(s)), _) => decoded(&s).map(Program::Sfnt),
                (_, _, Some(Object::Stream(s))) if open_type(&s)? => decoded(&s).map(Program::Sfnt),
                (_, _, Some(Object::Stream(s))) if bare_cff(&s)? => decoded(&s).map(Program::Cff),
                (Some(Object::Stream(s)), ..) => decoded(&s).map(Program::Type1),
                _ => None,
            },
        )
    }

    /// A composite font's descendant CID font (empty when it names none).
    fn cid_font(&self, dict: &Dict) -> Result<Dict> {
        Ok(match self.get(dict, b"DescendantFonts")? {
            Some(Object::Array(a)) => match a.first().map(|d| self.resolve(d)).transpose()? {
                Some(Object::Dict(d)) => d,
                _ => Dict::new(),
            },
            _ => Dict::new(),
        })
    }

    /// A CID font's `/CIDToGIDMap` as glyph ids by CID; None for `/Identity` or none given.
    fn cid_to_gid(&self, cid_font: &Dict) -> Result<Option<Vec<u16>>> {
        Ok(match self.get(cid_font, b"CIDToGIDMap")? {
            Some(Object::Stream(s)) => {
                self.decode(&s)
                    .ok()
                    .and_then(|d| d.into_bytes().ok())
                    .map(|data| {
                        data.as_chunks::<2>()
                            .0
                            .iter()
                            .map(|p| u16::from_be_bytes(*p))
                            .collect()
                    })
            }
            _ => None,
        })
    }

    /// The number `descriptor[key]` holds, resolved; None when there is no descriptor or no number.
    fn number(&self, descriptor: Option<&Dict>, key: &[u8]) -> Result<Option<f64>> {
        Ok(match descriptor {
            Some(d) => self.get(d, key)?.as_ref().and_then(Object::as_number),
            None => None,
        })
    }

    /// The descriptor's `/Flags`, resolved; 0 when absent.
    fn flags(&self, descriptor: Option<&Dict>) -> Result<i64> {
        Ok(match descriptor {
            Some(d) => self.get(d, b"Flags")?.as_ref().and_then(Object::as_integer),
            None => None,
        }
        .unwrap_or(0))
    }

    fn descriptor(&self, dict: &Dict) -> Result<Option<Dict>> {
        Ok(match self.get(dict, b"FontDescriptor")? {
            Some(Object::Dict(d)) => Some(d),
            _ => None,
        })
    }

    fn simple(
        &self,
        dict: &Dict,
        subtype: &str,
        name: &str,
        symbols: Option<Symbols>,
        descriptor: Option<&Dict>,
        program: Option<&Program>,
    ) -> Result<Kind> {
        let first = self
            .get(dict, b"FirstChar")?
            .and_then(|o| o.as_integer())
            .unwrap_or(0);
        let widths = match self.get(dict, b"Widths")? {
            Some(Object::Array(w)) => w
                .iter()
                .map(|v| {
                    self.resolve(v)
                        .ok()
                        .and_then(|o| o.as_number())
                        .unwrap_or(0.0)
                })
                .collect(),
            _ => Vec::new(),
        };
        let missing = self.number(descriptor, b"MissingWidth")?.unwrap_or(0.0);
        let standard = base14(name);
        let (encoding, names) =
            self.simple_encoding(dict, subtype, symbols, descriptor, program)?;
        Ok(Kind::Simple {
            first,
            widths,
            missing,
            encoding,
            names,
            base14: standard,
        })
    }

    /// A simple font's text and glyph names by code: its base encoding (named, built into its program, or by its
    /// kind) with its `/Differences` over it. A code's name is None where the program's own encoding selects its
    /// glyph: no base encoding is named and the font is a symbolic TrueType font, or embeds a Type 1 or CFF program
    /// (9.6.5.2: its built-in encoding is the base).
    fn simple_encoding(
        &self,
        dict: &Dict,
        subtype: &str,
        symbols: Option<Symbols>,
        descriptor: Option<&Dict>,
        program: Option<&Program>,
    ) -> Result<(ByCode, ByCode)> {
        let symbolic = self.flags(descriptor)? & 4 != 0;
        let (base_name, differences) = match self.get(dict, b"Encoding")? {
            Some(Object::Name(n)) => (Some(n), None),
            Some(Object::Dict(e)) => (
                self.get(&e, b"BaseEncoding")?
                    .and_then(|o| o.as_name().map(<[u8]>::to_vec)),
                match self.get(&e, b"Differences")? {
                    Some(Object::Array(a)) => Some(a),
                    _ => None,
                },
            ),
            _ => (None, None),
        };
        // The base encoding: the one named, else the font's own -- a symbol font's built in, Standard for any other
        // Type 1, and for a symbolic TrueType none at all (its codes are its glyphs).
        let table = |map: &pdf_encoding::ForwardMap| -> Vec<Option<String>> {
            (0..=255u8)
                .map(|b| map.get(b).map(|t| glyph_of_table(t).to_string()))
                .collect()
        };
        let mut encoding = match base_name.as_deref() {
            Some(b"WinAnsiEncoding") => table(&pdf_encoding::WINANSI),
            Some(b"MacRomanEncoding") => table(&pdf_encoding::MACROMAN),
            Some(b"MacExpertEncoding") => table(&pdf_encoding::MACEXPERT),
            Some(b"StandardEncoding") => table(&pdf_encoding::STANDARD),
            _ if subtype == "TrueType" && symbolic => vec![None; 256],
            _ if subtype == "TrueType" => table(&pdf_encoding::WINANSI),
            _ => match symbols {
                Some(symbols) => (0..=255u8)
                    .map(|b| symbols.character(b).map(String::from))
                    .collect(),
                None => table(&pdf_encoding::STANDARD),
            },
        };
        // The glyph names: the base encoding's characters' names; with no base named, none where the program's own
        // encoding is the base (a symbolic TrueType font, an embedded Type 1 or CFF program). Symbol and Zapf
        // Dingbats embedding no program are drawn by their built-in encodings (Annex D.5, D.6) whatever base is
        // named, as viewers draw them: a base named for one (`WinAnsiEncoding`) names glyphs it does not have. One
        // that embeds its program is drawn, and read, by the encoding named.
        let own_base = base_name.is_none()
            && ((subtype == "TrueType" && symbolic)
                || matches!(program, Some(Program::Type1(_) | Program::Cff(_))));
        let built_in = match symbols {
            Some(s @ (Symbols::Symbol | Symbols::ZapfDingbats)) if program.is_none() => Some(s),
            _ => None,
        };
        let name_of = |text: Option<String>| {
            let text = text?;
            let mut chars = text.chars();
            let (Some(c), None) = (chars.next(), chars.next()) else {
                return None;
            };
            let mut buffer = [0u8; 64];
            skrifa::raw::ps::agl::char_to_name(c, &mut buffer).map(str::to_string)
        };
        let mut names: Vec<Option<String>> = match built_in {
            Some(symbols) => (0..=255u8)
                .map(|b| name_of(symbols.character(b).map(String::from)))
                .collect(),
            None if own_base => vec![None; 256],
            None => encoding.iter().cloned().map(name_of).collect(),
        };
        // Its text follows the glyphs drawn (D121): Symbol's Greek and Zapf Dingbats' dingbats, as viewers show them,
        // not the Latin of a base encoding named for it.
        if let Some(symbols) = built_in {
            encoding = (0..=255u8)
                .map(|b| symbols.character(b).map(String::from))
                .collect();
        }
        // An embedded Type 1 program with no base encoding named: the program's own encoding is the base.
        if base_name.is_none()
            && matches!(subtype, "Type1" | "MMType1")
            && let Some(Program::Type1(program)) = program
            && let Some(builtin) = fontfile::type1_encoding(program)
        {
            for (slot, name) in encoding.iter_mut().zip(builtin) {
                *slot = name.and_then(|n| glyph_text(&n));
            }
        }
        if let Some(differences) = differences {
            let mut code = 0usize;
            for item in differences {
                match item {
                    Object::Integer(i) => code = usize::try_from(i).unwrap_or(256),
                    Object::Name(glyph) => {
                        if code < 256 {
                            let glyph = String::from_utf8_lossy(&glyph).into_owned();
                            encoding[code] = glyph_text(&glyph);
                            names[code] = Some(glyph);
                        }
                        code += 1;
                    }
                    _ => {}
                }
            }
        }
        Ok((encoding, names))
    }

    fn composite(&self, dict: &Dict, cid_font: &Dict) -> Result<Kind> {
        // The `CMap`, and the collection whose text its `CID`s stand for: a predefined `CMap`'s own (ISO 32000-2
        // section 9.10.2), Identity's and an embedded one's the `CIDFont`'s; none for a name no table defines, whose
        // codes' `CID`s are not known (they are read as Identity's). The writing mode is the `CMap`'s: a predefined
        // one's by its table, Identity's by its name, an embedded one's by its program's `/WMode`, else its
        // dictionary's.
        let (cmap, vertical, code_text, collection) = match self.get(dict, b"Encoding")? {
            Some(Object::Name(n)) => {
                let name = String::from_utf8_lossy(&n).into_owned();
                if let Some((cmap, vertical, collection)) = CMap::predefined(&name) {
                    // A Unicode `CMap`'s codes are the text; a legacy CJK one's are bytes of its encoding.
                    let code_text = if name.starts_with("Uni")
                        && (name.contains("UCS2") || name.contains("UTF16"))
                    {
                        CodeText::Utf16
                    } else {
                        legacy_encoding(&name).map_or(CodeText::Unknown, CodeText::Legacy)
                    };
                    (cmap, vertical, code_text, collection)
                } else {
                    let identity = matches!(name.as_str(), "Identity-H" | "Identity-V");
                    let collection = if identity {
                        self.collection(cid_font)?
                    } else {
                        None
                    };
                    let vertical = name.ends_with("-V");
                    (CMap::identity(), vertical, CodeText::Unknown, collection)
                }
            }
            Some(Object::Stream(s)) => {
                let cmap = CMap::parse(&self.decode(&s)?.into_bytes()?);
                let stated = match cmap.wmode() {
                    Some(mode) => Some(mode),
                    None => self.get(&s.dict, b"WMode")?.and_then(|o| o.as_integer()),
                };
                (
                    cmap,
                    stated == Some(1),
                    CodeText::Unknown,
                    self.collection(cid_font)?,
                )
            }
            _ => (CMap::identity(), false, CodeText::Unknown, None),
        };
        let width = self
            .get(cid_font, b"DW")?
            .and_then(|o| o.as_number())
            .unwrap_or(1000.0);
        let widths = self.cid_metric(cid_font, b"W", 1, 0, width)?;
        let vertical = if vertical {
            // `/DW2` is [vy w1y], [880 -1000] when absent.
            let dw2 = match self.get(cid_font, b"DW2")? {
                Some(Object::Array(dw2)) => dw2,
                _ => Vec::new(),
            };
            let stated =
                |i: usize, default: f64| dw2.get(i).and_then(Object::as_number).unwrap_or(default);
            Some(Box::new(Vertical {
                displacement: self.cid_metric(cid_font, b"W2", 3, 0, stated(1, -1000.0))?,
                across: self.cid_metric(cid_font, b"W2", 3, 1, 0.0)?,
                down: self.cid_metric(cid_font, b"W2", 3, 2, stated(0, 880.0))?,
            }))
        } else {
            None
        };
        Ok(Kind::Composite {
            cmap,
            widths,
            vertical,
            code_text,
            collection,
        })
    }

    /// The metric `key` of a `CIDFont` states, `default` for any `CID` it does not list: `/W`, one number a `CID`
    /// (its width w0), or `/W2`, three (w1y vx vy), of which the one at `index` is read. A list is `c [v ...]`, a
    /// run `c_first c_last v...`.
    fn cid_metric(
        &self,
        cid_font: &Dict,
        key: &[u8],
        per: usize,
        index: usize,
        default: f64,
    ) -> Result<CidMetric> {
        let mut runs = Vec::new();
        if let Some(Object::Array(array)) = self.get(cid_font, key)? {
            let items: Vec<Object> = array
                .iter()
                .map(|i| self.resolve(i).unwrap_or(Object::Null))
                .collect();
            let mut i = 0;
            while i + 1 < items.len() {
                let Some(first) = items[i].as_integer().and_then(|v| u32::try_from(v).ok()) else {
                    break;
                };
                match &items[i + 1] {
                    Object::Array(list) => {
                        for (cid, values) in (first..).zip(list.chunks(per)) {
                            let value = match values.get(index) {
                                Some(v) => self.resolve(v)?.as_number().unwrap_or(default),
                                None => default,
                            };
                            runs.push((cid, cid, value));
                        }
                        i += 2;
                    }
                    last => {
                        let (Some(last), Some(value)) = (
                            last.as_integer().and_then(|v| u32::try_from(v).ok()),
                            items.get(i + 2 + index).and_then(Object::as_number),
                        ) else {
                            break;
                        };
                        runs.push((first, last, value));
                        i += 2 + per;
                    }
                }
            }
        }
        Ok(CidMetric { runs, default })
    }

    /// Adobe's table of the collection a `CIDFont`'s `/CIDSystemInfo` names, when there is one.
    fn collection(&self, cid_font: &Dict) -> Result<Option<&'static Collection>> {
        let Some(Object::Dict(info)) = self.get(cid_font, b"CIDSystemInfo")? else {
            return Ok(None);
        };
        Ok(
            match (self.get(&info, b"Registry")?, self.get(&info, b"Ordering")?) {
                (Some(Object::String(registry)), Some(Object::String(ordering))) => {
                    Collection::named(&registry, &ordering)
                }
                _ => None,
            },
        )
    }
}

/// An embedded font program, decoded, by what this reader reads of it.
#[derive(Debug, Clone)]
enum Program {
    /// A Type 1 program (`/FontFile`): its clear-text part.
    Type1(Vec<u8>),
    /// A TrueType or OpenType program (`/FontFile2`, `/FontFile3` of subtype `OpenType`): its tables.
    Sfnt(Vec<u8>),
    /// A bare CFF program (`/FontFile3` of subtype `Type1C` or `CIDFontType0C`).
    Cff(Vec<u8>),
}

/// The weight class `program` states of itself, and the warning naming why it is unread when it does not read.
fn program_weight(program: Option<&Program>) -> (Option<f64>, Vec<String>) {
    match program.map(Program::weight) {
        Some(Err(e)) => (None, vec![format!("its program's weight unread: {e}")]),
        weight => (
            weight.and_then(std::result::Result::ok).flatten(),
            Vec::new(),
        ),
    }
}

impl Program {
    /// Its bytes, decoded.
    fn data(&self) -> &[u8] {
        match self {
            Self::Type1(data) | Self::Sfnt(data) | Self::Cff(data) => data,
        }
    }

    /// The PostScript name the program states of itself (a TrueType/OpenType program's; a Type 1's or a bare CFF's is
    /// not read).
    fn postscript_name(&self) -> Option<String> {
        match self {
            Self::Type1(_) | Self::Cff(_) => None,
            Self::Sfnt(data) => fontfile::sfnt_postscript_name(data),
        }
    }

    /// The weight class the program states of itself; an error names what makes it unreadable.
    fn weight(&self) -> std::result::Result<Option<f64>, String> {
        match self {
            Self::Type1(data) => Ok(fontfile::type1_weight(data)),
            Self::Sfnt(data) => Ok(fontfile::sfnt_weight(data)),
            Self::Cff(data) => fontfile::cff_weight(data).map_err(|e| format!("CFF: {e}")),
        }
    }
}

/// The text of a base-encoding table entry as its glyph's name says. The tables give Latin-1's characters where
/// ISO 32000 Annex D names the glyphs `space` (code 32, and `WinAnsiEncoding` octal 240) and `hyphen` (its octal 255):
/// U+00A0 and U+00AD; the Adobe Glyph List's `space` and `hyphen` are U+0020 and U+002D. Symbol's and Zapf Dingbats'
/// tables give Adobe's corporate-use code points for the glyphs Unicode had not encoded then: those it has since
/// encoded are their characters.
fn glyph_of_table(text: char) -> char {
    match text {
        '\u{A0}' => ' ',
        '\u{AD}' => '-',
        other => corporate_use(other).unwrap_or(other),
    }
}

/// The text a glyph name stands for: the Adobe Glyph List's (a corporate-use code point it gives for a glyph Unicode
/// has since encoded, as `parenlefttp`'s, read as that character), or a `uniXXXX` / `uXXXX[XX]` name's.
pub(crate) fn glyph_text(name: &str) -> Option<String> {
    if let Some(t) = pdf_encoding::glyphname_to_unicode(name) {
        let mut chars = t.chars();
        return Some(match (chars.next().and_then(corporate_use), chars.next()) {
            (Some(c), None) => c.to_string(),
            _ => t.to_string(),
        });
    }
    let base = name.split('.').next().unwrap_or(name);
    if let Some(hex) = base
        .strip_prefix("uni")
        .filter(|h| h.len() >= 4 && h.len() % 4 == 0)
    {
        let units: Option<Vec<u16>> = hex
            .as_bytes()
            .chunks(4)
            .map(|c| u16::from_str_radix(std::str::from_utf8(c).ok()?, 16).ok())
            .collect();
        return char::decode_utf16(units?)
            .collect::<std::result::Result<String, _>>()
            .ok();
    }
    if let Some(hex) = base
        .strip_prefix('u')
        .filter(|h| (4..=6).contains(&h.len()))
    {
        return u32::from_str_radix(hex, 16)
            .ok()
            .and_then(char::from_u32)
            .map(String::from);
    }
    (base != name).then(|| glyph_text(base)).flatten()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glyph_names_resolve_through_the_list_and_the_uni_forms() {
        assert_eq!(glyph_text("Aacute").as_deref(), Some("\u{C1}"));
        assert_eq!(glyph_text("uni03B1").as_deref(), Some("\u{3B1}"));
        assert_eq!(glyph_text("u1D70F").as_deref(), Some("\u{1D70F}"));
        assert_eq!(glyph_text("fi").as_deref(), Some("\u{FB01}"));
        assert_eq!(glyph_text("a.sc").as_deref(), Some("a"));
        assert_eq!(glyph_text("g123"), None);
    }

    #[test]
    fn standard_font_names_and_aliases_find_their_metrics() {
        assert_eq!(
            base14("ABCDEF+Helvetica-Bold").map(|m| m.name),
            Some("Helvetica-Bold")
        );
        assert_eq!(
            base14("TimesNewRomanPS-ItalicMT").map(|m| m.name),
            Some("Times-Italic")
        );
        assert_eq!(
            standard_width(base14("Helvetica").unwrap(), Some("A")),
            Some(667.0)
        );
        assert!(base14("CMR10").is_none());
    }

    #[test]
    fn a_standard_font_measures_its_ligatures() {
        let times = base14("Times-Roman").unwrap();
        assert_eq!(standard_width(times, Some("\u{FB01}")), Some(556.0));
        assert_eq!(standard_width(times, Some("\u{FB02}")), Some(556.0));
    }

    fn info(name: &str) -> FontInfo {
        FontInfo {
            name: name.to_string(),
            subtype: "Type1".to_string(),
            flags: 0,
            descriptor_weight: None,
            program_weight: None,
            italic_angle: None,
            embedded: false,
        }
    }

    #[test]
    fn a_subset_tag_is_six_capitals_and_a_plus() {
        assert_eq!(info("ABCDEF+Helvetica").name_without_subset(), "Helvetica");
        assert_eq!(
            info("Abcdef+Helvetica").name_without_subset(),
            "Abcdef+Helvetica"
        );
        assert_eq!(info("ABC+Helvetica").name_without_subset(), "ABC+Helvetica");
    }

    #[test]
    fn a_name_states_the_weight_of_its_style_part() {
        let weight = |name: &str| info(name).name_weight();
        assert_eq!(weight("ABCDEF+TimesNewRomanPS-BoldMT"), Some(700.0));
        assert_eq!(weight("Arial,BoldItalic"), Some(700.0));
        assert_eq!(weight("SourceSansPro-SemiBold"), Some(600.0));
        assert_eq!(weight("AvantGarde-Demi"), Some(600.0));
        assert_eq!(weight("NotoSansCJKjp-DemiLight"), Some(350.0));
        assert_eq!(weight("HelveticaNeueLTStd-BdCn"), Some(700.0));
        assert_eq!(weight("LMRoman10-Bold"), Some(700.0));
        assert_eq!(weight("ArialBlack"), Some(900.0));
        assert_eq!(weight("Times-Roman"), Some(400.0));
        // The family's own words state nothing, and a name with no style part states no weight.
        assert_eq!(weight("BlackChancery"), None);
        assert_eq!(weight("Boldface-Regular"), Some(400.0));
        assert_eq!(weight("CMBX10"), None);
    }

    #[test]
    fn any_statement_of_a_heavy_weight_makes_a_font_bold() {
        assert!(info("Helvetica-Bold").bold());
        assert!(!info("Helvetica").bold());
        assert!(!info("MinionPro-Medium").bold());
        assert!(
            FontInfo {
                flags: 1 << 18,
                ..info("F1")
            }
            .bold()
        );
        assert!(
            FontInfo {
                descriptor_weight: Some(600.0),
                ..info("F1")
            }
            .bold()
        );
        assert!(
            FontInfo {
                program_weight: Some(700.0),
                ..info("CMBX10")
            }
            .bold()
        );
        // A regular program shown under a bold name is drawn emboldened: the name's statement stands.
        assert!(
            FontInfo {
                program_weight: Some(400.0),
                ..info("Calibri,Bold")
            }
            .bold()
        );
    }

    #[test]
    fn a_code_reads_as_unicode_or_as_its_legacy_encoding() {
        assert_eq!(
            legacy_text(CodeText::Legacy(encoding_rs::GBK), 0xC4E3, 2).as_deref(),
            Some("\u{4F60}")
        );
        assert_eq!(
            legacy_text(CodeText::Legacy(encoding_rs::SHIFT_JIS), 0x82A0, 2).as_deref(),
            Some("\u{3042}")
        );
        assert_eq!(
            unicode_text(CodeText::Utf16, 0x3B1, 2).as_deref(),
            Some("\u{3B1}")
        );
        assert_eq!(
            unicode_text(CodeText::Utf16, 0xD840_DC0B, 4).as_deref(),
            Some("\u{2000B}")
        );
        assert_eq!(
            unicode_text(CodeText::Legacy(encoding_rs::GBK), 0x41, 1),
            None
        );
        assert_eq!(legacy_text(CodeText::Unknown, 0x41, 2), None);
    }
}
