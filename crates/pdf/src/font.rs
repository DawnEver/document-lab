//! Fonts (ISO 32000-2 section 9.5-9.10), read for what text needs: how a string splits into codes, each code's
//! width and Unicode text, the font's ascent and descent, and whether it is bold ([`FontInfo::bold`]). Glyph
//! outlines are not read.

use crate::base14::{FONTS, Metrics};
use crate::cid2unicode::Collection;
use crate::cmap::{CMap, legacy_encoding};
use crate::error::Result;
use crate::file::File;
use crate::fontfile::{self, GlyphText};
use crate::geometry::Matrix;
use crate::object::{Dict, Object, Stream};

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
    /// `/Weight`. A bare CFF program's (`/FontFile3` `Type1C`, `CIDFontType0C`) is not read.
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
        style_weight(words(self.name_without_subset()).into_iter().skip(1))
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
    /// The facts of the font named `name`, of `subtype`, its descriptor (its values resolved in `file`) and
    /// embedded program stating them.
    ///
    /// Refuses a reference in the descriptor that does not resolve.
    fn new(
        file: &File,
        name: String,
        subtype: String,
        descriptor: Option<&Dict>,
        program: Option<&Program>,
    ) -> Result<Self> {
        Ok(Self {
            name,
            subtype,
            flags: file.flags(descriptor)?,
            descriptor_weight: file.number(descriptor, b"FontWeight")?,
            program_weight: program.and_then(Program::weight),
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

/// How codes become glyphs.
#[derive(Debug, Clone)]
enum Kind {
    /// One byte per code; widths by code; Unicode by the encoding.
    Simple {
        first: i64,
        widths: Vec<f64>,
        missing: f64,
        encoding: Vec<Option<String>>,
        base14: Option<&'static Metrics>,
    },
    /// Codes by a `CMap`, widths by `CID`; vertical displacements by `CID` when the `CMap` writes vertically.
    Composite {
        cmap: CMap,
        widths: CidMetric,
        vertical: Option<CidMetric>,
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
struct CidMetric {
    /// (first `CID`, last `CID`, value) runs, in the order stated: the first run holding a `CID` gives its value.
    runs: Vec<(u32, u32, f64)>,
    /// The value of a `CID` no run holds.
    default: f64,
}

impl CidMetric {
    fn of(&self, cid: u32) -> f64 {
        self.runs
            .iter()
            .find(|(lo, hi, _)| (*lo..=*hi).contains(&cid))
            .map_or(self.default, |run| run.2)
    }
}

/// What a composite font's codes are, as text, when no `ToUnicode` `CMap` says.
#[derive(Debug, Clone, Copy)]
enum CodeText {
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
    pub info: FontInfo,
    kind: Kind,
    to_unicode: Option<CMap>,
    /// The embedded TrueType program's glyph text, the last word on a glyph's text.
    program: Option<GlyphText>,
    /// A `CIDFontType2`'s CID-to-glyph map (None: the identity).
    cid_to_gid: Option<Vec<u16>>,
    /// From glyph space to text space (`[0.001 0 0 0.001 0 0]` but for Type 3 fonts).
    pub matrix: Matrix,
    /// The ascent in glyph space.
    pub ascent: f64,
    /// The descent in glyph space (negative).
    pub descent: f64,
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
}

const GLYPH_SPACE: Matrix = Matrix([0.001, 0.0, 0.0, 0.001, 0.0, 0.0]);

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
                            let program = self.program.as_ref()?;
                            program.text(program.glyph_of_code(code)?)
                        });
                    let stated = usize::try_from(i64::from(b) - first)
                        .ok()
                        .and_then(|i| widths.get(i))
                        .copied();
                    let width = stated
                        .or_else(|| base14.and_then(|m| standard_width(m, text.as_deref())))
                        .unwrap_or(*missing);
                    Code {
                        code,
                        bytes: 1,
                        text,
                        width: width * units,
                        vertical_advance: None,
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
                            self.program.as_ref()?.text(gid)
                        })
                        .or_else(|| collection.and_then(|c| c.text(cid)))
                        .or_else(|| legacy_text(*code_text, code, len));
                    Code {
                        code,
                        bytes: len,
                        text,
                        width: widths.of(cid) * units,
                        vertical_advance: vertical.as_ref().map(|v| v.of(cid) * units),
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
        let kind = match &cid_font {
            Some(cid_font) => self.composite(dict, cid_font)?,
            None => self.simple(
                dict,
                &subtype,
                &base_name,
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
        let number = |key: &[u8]| self.number(descriptor.as_ref(), key);
        let standard = base14(&base_name);
        let bbox = match &descriptor {
            Some(d) => self.rect(d, b"FontBBox")?,
            None if subtype == "Type3" => self.rect(dict, b"FontBBox")?,
            None => None,
        };
        // Ascent and descent: the descriptor's, else the standard font's, else the font box's, else 0.8 and -0.2 of
        // the size (the proportions of a Latin text face).
        // (In glyph space: a Type 3 font's is its own, through its matrix.)
        let unit = 0.001 / matrix.0[3].abs().max(f64::MIN_POSITIVE);
        let ascent = number(b"Ascent")?
            .filter(|a| *a > 0.0)
            .or(standard.map(|m| m.ascent))
            .or(bbox.map(|b| b.y1))
            .unwrap_or(800.0 * unit);
        let descent = number(b"Descent")?
            .filter(|d| *d < 0.0)
            .or(standard.map(|m| m.descent))
            .or(bbox.map(|b| b.y0))
            .unwrap_or(-200.0 * unit);
        let cid_to_gid = match &cid_font {
            Some(cid_font) => self.cid_to_gid(cid_font)?,
            None => None,
        };
        Ok(Font {
            info: FontInfo::new(
                self,
                base_name,
                subtype,
                descriptor.as_ref(),
                program.as_ref(),
            )?,
            kind,
            to_unicode,
            program: match program {
                Some(Program::Sfnt(data)) => Some(GlyphText::parse(&data)),
                _ => None,
            },
            cid_to_gid,
            matrix,
            ascent,
            descent,
        })
    }

    /// The program `descriptor` embeds, decoded; None when it embeds none, or one that does not decode.
    fn program(&self, descriptor: Option<&Dict>) -> Result<Option<Program>> {
        let Some(descriptor) = descriptor else {
            return Ok(None);
        };
        let decoded = |stream: &Stream| self.decode(stream).ok().and_then(|d| d.into_bytes().ok());
        let open_type = |stream: &Stream| -> Result<bool> {
            Ok(self
                .get(&stream.dict, b"Subtype")?
                .as_ref()
                .and_then(Object::as_name)
                == Some(b"OpenType"))
        };
        Ok(
            match (
                self.get(descriptor, b"FontFile")?,
                self.get(descriptor, b"FontFile2")?,
                self.get(descriptor, b"FontFile3")?,
            ) {
                (_, Some(Object::Stream(s)), _) => decoded(&s).map(Program::Sfnt),
                (_, _, Some(Object::Stream(s))) if open_type(&s)? => decoded(&s).map(Program::Sfnt),
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
        let encoding = self.simple_encoding(dict, subtype, standard, descriptor, program)?;
        Ok(Kind::Simple {
            first,
            widths,
            missing,
            encoding,
            base14: standard,
        })
    }

    /// A simple font's text by code: its base encoding (named, built into its program, or by its kind) with its
    /// `/Differences` over it.
    fn simple_encoding(
        &self,
        dict: &Dict,
        subtype: &str,
        standard: Option<&'static Metrics>,
        descriptor: Option<&Dict>,
        program: Option<&Program>,
    ) -> Result<Vec<Option<String>>> {
        let symbolic = self.flags(descriptor)? & 4 != 0;
        // The base encoding: the one named, else the font's own -- Symbol's and ZapfDingbats' built in, Standard
        // for any other Type 1, and for a symbolic TrueType none at all (its codes are its glyphs).
        let builtin: &pdf_encoding::ForwardMap = match standard.map(|m| m.name) {
            Some("Symbol") => &pdf_encoding::SYMBOL,
            Some("ZapfDingbats") => &pdf_encoding::ZDINGBAT,
            _ => &pdf_encoding::STANDARD,
        };
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
        let base: Option<&pdf_encoding::ForwardMap> = match base_name.as_deref() {
            Some(b"WinAnsiEncoding") => Some(&pdf_encoding::WINANSI),
            Some(b"MacRomanEncoding") => Some(&pdf_encoding::MACROMAN),
            Some(b"MacExpertEncoding") => Some(&pdf_encoding::MACEXPERT),
            Some(b"StandardEncoding") => Some(&pdf_encoding::STANDARD),
            _ if subtype == "TrueType" && symbolic => None,
            _ if subtype == "TrueType" => Some(&pdf_encoding::WINANSI),
            _ => Some(builtin),
        };
        let mut encoding: Vec<Option<String>> = (0..=255u8)
            .map(|b| {
                base.and_then(|m| m.get(b))
                    .map(|t| glyph_of_table(t).to_string())
            })
            .collect();
        // An embedded Type 1 program with no base encoding named: the program's own encoding is the base.
        if base_name.is_none()
            && matches!(subtype, "Type1" | "MMType1")
            && let Some(Program::Type1(program)) = program
            && let Some(names) = fontfile::type1_encoding(program)
        {
            for (slot, name) in encoding.iter_mut().zip(names) {
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
                            encoding[code] = glyph_text(&String::from_utf8_lossy(&glyph));
                        }
                        code += 1;
                    }
                    _ => {}
                }
            }
        }
        Ok(encoding)
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
        let widths = self.cid_metric(cid_font, b"W", 1, width)?;
        let vertical = if vertical {
            // `/DW2` is [vy w1y], [880 -1000] when absent.
            let displacement = match self.get(cid_font, b"DW2")? {
                Some(Object::Array(dw2)) => dw2.get(1).and_then(Object::as_number),
                _ => None,
            }
            .unwrap_or(-1000.0);
            Some(self.cid_metric(cid_font, b"W2", 3, displacement)?)
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
    /// (its width w0), or `/W2`, three (w1y vx vy), of which the displacement w1y is read. (The position vector
    /// (vx, vy) places the glyph's outline off its vertical origin; text is placed by the origin and the
    /// displacement, so it is not read.) A list is `c [v ...]`, a run `c_first c_last v`.
    fn cid_metric(
        &self,
        cid_font: &Dict,
        key: &[u8],
        per: usize,
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
                            let value = self.resolve(&values[0])?.as_number().unwrap_or(default);
                            runs.push((cid, cid, value));
                        }
                        i += 2;
                    }
                    last => {
                        let (Some(last), Some(value)) = (
                            last.as_integer().and_then(|v| u32::try_from(v).ok()),
                            items.get(i + 2).and_then(Object::as_number),
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
}

impl Program {
    /// The weight class the program states of itself.
    fn weight(&self) -> Option<f64> {
        match self {
            Self::Type1(data) => fontfile::type1_weight(data),
            Self::Sfnt(data) => fontfile::sfnt_weight(data),
        }
    }
}

/// The text of a base-encoding table entry as its glyph's name says. The tables give Latin-1's characters where
/// ISO 32000 Annex D names the glyphs `space` (code 32, and `WinAnsiEncoding` octal 240) and `hyphen` (its octal 255):
/// U+00A0 and U+00AD; the Adobe Glyph List's `space` and `hyphen` are U+0020 and U+002D.
fn glyph_of_table(text: char) -> char {
    match text {
        '\u{A0}' => ' ',
        '\u{AD}' => '-',
        other => other,
    }
}

/// The text a glyph name stands for: the Adobe Glyph List's, or a `uniXXXX` / `uXXXX[XX]` name's.
pub(crate) fn glyph_text(name: &str) -> Option<String> {
    if let Some(t) = pdf_encoding::glyphname_to_unicode(name) {
        return Some(t.to_string());
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
