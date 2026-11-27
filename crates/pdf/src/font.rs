//! Fonts (ISO 32000-2 section 9.5-9.10), read for what text needs: how a string splits into codes, each code's
//! width and Unicode text, and the font's ascent and descent. Glyph outlines are not read.

use crate::base14::{FONTS, Metrics};
use crate::cmap::CMap;
use crate::error::Result;
use crate::file::File;
use crate::fontfile::GlyphText;
use crate::geometry::Matrix;
use crate::object::{Dict, Object};

/// What identifies a font to a reader: its name and the descriptor's facts.
#[derive(Debug, Clone, PartialEq)]
pub struct FontInfo {
    /// `/BaseFont` (a subset prefix such as `ABCDEF+` kept), or `/Name` for a Type 3 font.
    pub name: String,
    /// `/Subtype`: `Type1`, `TrueType`, `Type0`, `Type3`, `MMType1`.
    pub subtype: String,
    /// The descriptor's `/Flags` (bit 19 `ForceBold`, bit 7 `Italic`, ...); 0 when absent.
    pub flags: i64,
    /// The descriptor's `/FontWeight` (100-900), when stated.
    pub weight: Option<f64>,
    /// The descriptor's `/ItalicAngle` (degrees), when stated.
    pub italic_angle: Option<f64>,
    /// Whether the font program is embedded.
    pub embedded: bool,
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
    let name = name.split_once('+').map_or(name, |(_, rest)| rest);
    let canonical = match name.replace(' ', "").as_str() {
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
        let (descriptor, kind, vertical) = if subtype == "Type0" {
            self.composite(dict)?
        } else {
            self.simple(dict, &subtype, &base_name)?
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
        let program = self.program(descriptor.as_ref())?;
        let cid_to_gid = if subtype == "Type0" {
            self.cid_to_gid(dict)?
        } else {
            None
        };
        let embedded = descriptor.as_ref().is_some_and(|d| {
            [&b"FontFile"[..], b"FontFile2", b"FontFile3"]
                .iter()
                .any(|k| d.get(k).is_some())
        });
        Ok(Font {
            info: FontInfo {
                name: base_name,
                subtype,
                flags: descriptor
                    .as_ref()
                    .and_then(|d| d.get(b"Flags"))
                    .and_then(Object::as_integer)
                    .unwrap_or(0),
                weight: number(b"FontWeight"),
                italic_angle: number(b"ItalicAngle"),
                embedded,
            },
            kind,
            to_unicode,
            program,
            cid_to_gid,
            matrix,
            ascent,
            descent,
            vertical,
        })
    }

    /// The glyph text of the TrueType/OpenType program `descriptor` embeds, if it embeds one.
    fn program(&self, descriptor: Option<&Dict>) -> Result<Option<GlyphText>> {
        let Some(descriptor) = descriptor else {
            return Ok(None);
        };
        let stream = match self.get(descriptor, b"FontFile2")? {
            Some(Object::Stream(s)) => s,
            _ => match self.get(descriptor, b"FontFile3")? {
                Some(Object::Stream(s))
                    if s.dict.get(b"Subtype").and_then(Object::as_name) == Some(b"OpenType") =>
                {
                    s
                }
                _ => return Ok(None),
            },
        };
        Ok(self
            .decode(&stream)
            .ok()
            .and_then(|d| d.into_bytes().ok())
            .map(|data| GlyphText::parse(&data)))
    }

    /// A composite font's `/CIDToGIDMap` as glyph ids by CID; None for `/Identity` or none given.
    fn cid_to_gid(&self, dict: &Dict) -> Result<Option<Vec<u16>>> {
        let Some(Object::Array(descendants)) = self.get(dict, b"DescendantFonts")? else {
            return Ok(None);
        };
        let Some(Object::Dict(cid_font)) =
            descendants.first().map(|d| self.resolve(d)).transpose()?
        else {
            return Ok(None);
        };
        Ok(match self.get(&cid_font, b"CIDToGIDMap")? {
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

    fn simple(&self, dict: &Dict, subtype: &str, name: &str) -> Result<(Option<Dict>, Kind, bool)> {
        let descriptor = self.descriptor(dict)?;
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
            .as_ref()
            .and_then(|d| d.get(b"MissingWidth"))
            .and_then(Object::as_number)
            .unwrap_or(0.0);
        let standard = base14(name);
        let encoding = self.simple_encoding(dict, subtype, standard, descriptor.as_ref())?;
        Ok((
            descriptor,
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
            && let Some(Object::Stream(program)) = descriptor
                .map(|d| self.get(d, b"FontFile"))
                .transpose()?
                .flatten()
            && let Some(names) = self
                .decode(&program)
                .ok()
                .and_then(|d| d.into_bytes().ok())
                .and_then(|d| crate::fontfile::type1_encoding(&d))
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

    fn composite(&self, dict: &Dict) -> Result<(Option<Dict>, Kind, bool)> {
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
        let descendant = match self.get(dict, b"DescendantFonts")? {
            Some(Object::Array(a)) => match a.first().map(|d| self.resolve(d)).transpose()? {
                Some(Object::Dict(d)) => d,
                _ => Dict::new(),
            },
            _ => Dict::new(),
        };
        let default = self
            .get(&descendant, b"DW")?
            .and_then(|o| o.as_number())
            .unwrap_or(1000.0);
        let mut widths = Vec::new();
        if let Some(Object::Array(w)) = self.get(&descendant, b"W")? {
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
            self.descriptor(&descendant)?,
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
