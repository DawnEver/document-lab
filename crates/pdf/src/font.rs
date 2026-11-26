//! Fonts (ISO 32000-2 section 9.5-9.10), read for what text needs: how a string splits into codes, each code's
//! width and Unicode text, the font's ascent and descent, and whether it is bold ([`FontInfo::bold`]). Glyph
//! outlines are not read.

use crate::base14::{FONTS, Metrics};
use crate::cmap::CMap;
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
    /// The facts of the font named `name`, of `subtype`, its descriptor stating them and its embedded program
    /// stating `program_weight`.
    fn new(
        name: String,
        subtype: String,
        descriptor: Option<&Dict>,
        program_weight: Option<f64>,
    ) -> Self {
        let number = |key: &[u8]| {
            descriptor
                .and_then(|d| d.get(key))
                .and_then(Object::as_number)
        };
        Self {
            name,
            subtype,
            flags: descriptor
                .and_then(|d| d.get(b"Flags"))
                .and_then(Object::as_integer)
                .unwrap_or(0),
            descriptor_weight: number(b"FontWeight"),
            program_weight,
            italic_angle: number(b"ItalicAngle"),
            embedded: descriptor.is_some_and(|d| {
                [&b"FontFile"[..], b"FontFile2", b"FontFile3"]
                    .iter()
                    .any(|k| d.get(k).is_some())
            }),
        }
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
    /// Codes by a `CMap`, widths by `CID`.
    Composite {
        cmap: CMap,
        widths: Vec<(u32, u32, f64)>,
        default: f64,
        code_text: CodeText,
    },
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
    /// Whether the writing mode is vertical (read as horizontal; the text is kept).
    pub vertical: bool,
    /// What of it could not be read, in words (a program too malformed to state its weight); empty when all was.
    pub warnings: Vec<String>,
}

/// One code of a string: its value, Unicode text and width in text space (before the font size).
#[derive(Debug, Clone, PartialEq)]
pub struct Code {
    /// The code.
    pub code: u32,
    /// How many bytes it took.
    pub bytes: usize,
    /// Its text, when the font says what it is.
    pub text: Option<String>,
    /// Its horizontal advance in text space units of one font size.
    pub width: f64,
}

const GLYPH_SPACE: Matrix = Matrix([0.001, 0.0, 0.0, 0.001, 0.0, 0.0]);

impl Font {
    /// Split `bytes` into codes, each with its text and width.
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
                    }
                })
                .collect(),
            Kind::Composite {
                cmap,
                widths,
                default,
                code_text,
            } => cmap
                .codes(bytes)
                .into_iter()
                .map(|(code, len)| {
                    let cid = cmap.cid(code);
                    let text = self
                        .to_unicode
                        .as_ref()
                        .and_then(|c| c.text(code))
                        .map(str::to_string)
                        .or_else(|| code_as_text(*code_text, code, len))
                        .or_else(|| {
                            let gid = match &self.cid_to_gid {
                                Some(map) => *map.get(usize::try_from(cid).ok()?)?,
                                None => u16::try_from(cid).ok()?,
                            };
                            self.program.as_ref()?.text(gid)
                        });
                    let width = widths
                        .iter()
                        .find(|(lo, hi, _)| (*lo..=*hi).contains(&cid))
                        .map_or(*default, |w| w.2);
                    Code {
                        code,
                        bytes: len,
                        text,
                        width: width * units,
                    }
                })
                .collect(),
        }
    }
}

/// A composite font's code as text, by what its codes are.
fn code_as_text(kind: CodeText, code: u32, len: usize) -> Option<String> {
    match kind {
        CodeText::Unknown => None,
        CodeText::Utf16 => char::from_u32(code).map(String::from),
        CodeText::Legacy(encoding) => {
            let bytes = &code.to_be_bytes()[4 - len.min(4)..];
            let (text, _, failed) = encoding.decode(bytes);
            (!failed).then(|| text.into_owned())
        }
    }
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
        let name_of = |key: &[u8]| {
            dict.get(key)
                .and_then(Object::as_name)
                .map(|n| String::from_utf8_lossy(n).into_owned())
        };
        let subtype = name_of(b"Subtype").unwrap_or_default();
        let base_name = name_of(b"BaseFont")
            .or_else(|| name_of(b"Name"))
            .unwrap_or_default();
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
        let (kind, vertical) = match &cid_font {
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
        let number = |key: &[u8]| {
            descriptor
                .as_ref()
                .and_then(|d| d.get(key))
                .and_then(Object::as_number)
        };
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
        let ascent = number(b"Ascent")
            .filter(|a| *a > 0.0)
            .or(standard.map(|m| m.ascent))
            .or(bbox.map(|b| b.y1))
            .unwrap_or(800.0 * unit);
        let descent = number(b"Descent")
            .filter(|d| *d < 0.0)
            .or(standard.map(|m| m.descent))
            .or(bbox.map(|b| b.y0))
            .unwrap_or(-200.0 * unit);
        let cid_to_gid = match &cid_font {
            Some(cid_font) => self.cid_to_gid(cid_font)?,
            None => None,
        };
        let (program_weight, warnings) = match program.as_ref().map(Program::weight) {
            Some(Err(e)) => (None, vec![format!("its program's weight unread: {e}")]),
            weight => (
                weight.and_then(std::result::Result::ok).flatten(),
                Vec::new(),
            ),
        };
        Ok(Font {
            info: FontInfo::new(base_name, subtype, descriptor.as_ref(), program_weight),
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
            vertical,
            warnings,
        })
    }

    /// The program `descriptor` embeds, decoded; None when it embeds none, or one that does not decode.
    fn program(&self, descriptor: Option<&Dict>) -> Result<Option<Program>> {
        let Some(descriptor) = descriptor else {
            return Ok(None);
        };
        let decoded = |stream: &Stream| self.decode(stream).ok().and_then(|d| d.into_bytes().ok());
        Ok(
            match (
                self.get(descriptor, b"FontFile")?,
                self.get(descriptor, b"FontFile2")?,
                self.get(descriptor, b"FontFile3")?,
            ) {
                (_, Some(Object::Stream(s)), _) => decoded(&s).map(Program::Sfnt),
                (_, _, Some(Object::Stream(s)))
                    if s.dict.get(b"Subtype").and_then(Object::as_name) == Some(b"OpenType") =>
                {
                    decoded(&s).map(Program::Sfnt)
                }
                (_, _, Some(Object::Stream(s)))
                    if matches!(
                        s.dict.get(b"Subtype").and_then(Object::as_name),
                        Some(b"Type1C" | b"CIDFontType0C")
                    ) =>
                {
                    decoded(&s).map(Program::Cff)
                }
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
    ) -> Result<(Kind, bool)> {
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
        let missing = descriptor
            .and_then(|d| d.get(b"MissingWidth"))
            .and_then(Object::as_number)
            .unwrap_or(0.0);
        let standard = base14(name);
        let encoding = self.simple_encoding(dict, subtype, standard, descriptor, program)?;
        Ok((
            Kind::Simple {
                first,
                widths,
                missing,
                encoding,
                base14: standard,
            },
            false,
        ))
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
        let symbolic = descriptor
            .and_then(|d| d.get(b"Flags"))
            .and_then(Object::as_integer)
            .is_some_and(|f| f & 4 != 0);
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

    fn composite(&self, dict: &Dict, cid_font: &Dict) -> Result<(Kind, bool)> {
        let (cmap, vertical, code_text) = match self.get(dict, b"Encoding")? {
            Some(Object::Name(n)) => {
                let name = String::from_utf8_lossy(&n).into_owned();
                // A predefined `CMap`: Identity is read; a Unicode one (`Uni...-UCS2-` / `-UTF16-`) has the text as
                // its codes; a legacy CJK one has its codes as bytes of that encoding; any other selects CIDs from a
                // registry's table this reader does not carry.
                let vertical = name.ends_with("-V");
                if let Some((cmap, encoding)) = CMap::legacy(&name) {
                    (cmap, vertical, CodeText::Legacy(encoding))
                } else if name.starts_with("Uni")
                    && (name.contains("UCS2") || name.contains("UTF16"))
                {
                    (CMap::identity(), vertical, CodeText::Utf16)
                } else {
                    (CMap::identity(), vertical, CodeText::Unknown)
                }
            }
            Some(Object::Stream(s)) => {
                let vertical = self.get(&s.dict, b"WMode")?.and_then(|o| o.as_integer()) == Some(1);
                (
                    CMap::parse(&self.decode(&s)?.into_bytes()?),
                    vertical,
                    CodeText::Unknown,
                )
            }
            _ => (CMap::identity(), false, CodeText::Unknown),
        };
        let default = self
            .get(cid_font, b"DW")?
            .and_then(|o| o.as_number())
            .unwrap_or(1000.0);
        let mut widths = Vec::new();
        if let Some(Object::Array(w)) = self.get(cid_font, b"W")? {
            let items: Vec<Object> = w
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
                        for (cid, width) in (first..).zip(list.iter()) {
                            widths.push((
                                cid,
                                cid,
                                self.resolve(width)?.as_number().unwrap_or(default),
                            ));
                        }
                        i += 2;
                    }
                    last => {
                        let (Some(last), Some(width)) = (
                            last.as_integer().and_then(|v| u32::try_from(v).ok()),
                            items.get(i + 2).and_then(Object::as_number),
                        ) else {
                            break;
                        };
                        widths.push((first, last, width));
                        i += 3;
                    }
                }
            }
        }
        Ok((
            Kind::Composite {
                cmap,
                widths,
                default,
                code_text,
            },
            vertical,
        ))
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

impl Program {
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
    fn a_legacy_code_decodes_as_its_encoding() {
        assert_eq!(
            code_as_text(CodeText::Legacy(encoding_rs::GBK), 0xC4E3, 2).as_deref(),
            Some("\u{4F60}")
        );
        assert_eq!(
            code_as_text(CodeText::Legacy(encoding_rs::SHIFT_JIS), 0x82A0, 2).as_deref(),
            Some("\u{3042}")
        );
        assert_eq!(
            code_as_text(CodeText::Utf16, 0x3B1, 2).as_deref(),
            Some("\u{3B1}")
        );
        assert_eq!(code_as_text(CodeText::Unknown, 0x41, 2), None);
    }
}
