//! A glyph's shape (ISO 32000-2 sections 9.6-9.7): the outline its font program gives the glyph a character code
//! selects, in text space -- an em is 1, y up -- ready for the text rendering matrix.
//!
//! The program is the one the font embeds (`/FontFile` Type 1, `/FontFile2` TrueType, `/FontFile3` CFF, CID-keyed
//! CFF or OpenType), read by `skrifa` (MIT/Apache-2.0, pure Rust: TrueType, CFF, Type 1 with its eexec section). A
//! simple font that embeds none and names a standard font, or a common alias of one, is drawn with that font's
//! substitute, `PDFium`'s Foxit fonts (BSD-3-Clause, `src/fonts`); any other simple font that embeds none is drawn
//! with the Foxit font its descriptor's flags choose (fixed pitch, serif, else sans; bold, italic), each glyph
//! stretched across the width the file states for it, as `MuPDF` draws a substituted font. A symbol font other
//! than Symbol and Zapf Dingbats, and a composite font, that embeds none has no substitute: a CJK composite font
//! (its `CID`s of one of Adobe's CJK collections) and a Microsoft symbol font (Wingdings, Webdings) are drawn with
//! redistributable fallback programs not shipped here ([`FALLBACKS`]), each fetched by name when a glyph first
//! needs it (the file's [`Fetch`], [`crate::File::fetch_fallbacks`]) and reached through each glyph's character --
//! a composite font's glyphs at the fallback's own advance, as `MuPDF` sets its own, or stretched across the widths
//! the file states ([`FallbackWidths`], [`crate::File::set_fallback_widths`]). A
//! glyph none can draw -- any other font's, a fallback that cannot be had, a character none holds -- is
//! [`Shape::Missing`], by name, never guessed. A Type 3 glyph is its procedure (`/CharProcs`), which the
//! interpreter runs.
//!
//! Which glyph a code selects is the font's logic, not the program's (9.6.5, 9.7.4.2): in a simple font, the glyph
//! its encoding names -- by name in a Type 1 or CFF program, then by another name of the same character, by the
//! program's own encoding, by its Unicode `cmap`; in a TrueType program through the `cmap` subtable 9.6.5.4 says
//! (the name's Unicode in (3,1), its Mac OS Roman code in (1,0), else the code in (3,0) or (1,0)), then by its `post`
//! name -- and in a composite font the glyph its `CID` selects (`/CIDToGIDMap`, a CID-keyed CFF program's charset,
//! else the `CID` itself). A code that selects no glyph draws the program's glyph 0 (`.notdef`), as viewers do.

use std::cell::{OnceCell, RefCell};
use std::collections::HashMap;
use std::rc::Rc;

use skrifa::GlyphId;
use skrifa::font::charmap::EncodingKind;
use skrifa::font::glyph::outline::{DrawMemory, DrawOptions, PathSink};
use skrifa::raw::ps::agl;

use crate::font::{Font, Kind, Symbols, glyph_text};
use crate::geometry::{Point, Segment};
use crate::object::{Dict, Stream};

/// What a glyph is drawn from.
#[derive(Debug, Clone, PartialEq)]
pub enum Shape {
    /// Its outline in text space (an em is 1, y up), filled by the nonzero winding rule; empty for a glyph that
    /// draws nothing (a space).
    Outline(Rc<[Segment]>),
    /// A Type 3 glyph: its procedure draws it ([`crate::File::drawing`] runs it in place).
    Procedure,
    /// It cannot be drawn as the file states it: why, in words.
    Missing(Rc<str>),
}

/// Where a font's glyph shapes come from.
#[derive(Debug, Clone)]
pub(crate) enum Source {
    /// The program it embeds.
    Embedded(Rc<[u8]>),
    /// A standard font's substitute, for a font that embeds none; `stretched` when it stands for a font other than
    /// a standard one, its glyphs then stretched across the widths the file states.
    Substitute {
        substitute: &'static Substitute,
        stretched: bool,
    },
    /// A Type 3 font's glyph procedures (`/CharProcs`), and its `/Resources`, if any.
    Procedures {
        procedures: Dict,
        resources: Option<Dict>,
    },
    /// Fallback programs fetched when first needed ([`FALLBACKS`]), tried in turn for each glyph's character;
    /// `stretched`: its glyphs stretched across the widths the file states.
    Fallback {
        fallbacks: Fallbacks,
        chain: &'static [&'static str],
        stretched: bool,
    },
    /// None: why, in words.
    Unavailable(Rc<str>),
}

/// A substitute font program.
#[derive(Debug)]
pub(crate) struct Substitute {
    /// Its name.
    pub(crate) name: &'static str,
    data: &'static [u8],
}

macro_rules! substitutes {
    ($($constant:ident: $file:literal),* $(,)?) => {
        $(static $constant: Substitute = Substitute {
            name: $file,
            data: include_bytes!(concat!("fonts/", $file, ".cff")),
        };)*
    };
}

substitutes! {
    SANS: "FoxitSans",
    SANS_BOLD: "FoxitSansBold",
    SANS_ITALIC: "FoxitSansItalic",
    SANS_BOLD_ITALIC: "FoxitSansBoldItalic",
    SERIF: "FoxitSerif",
    SERIF_BOLD: "FoxitSerifBold",
    SERIF_ITALIC: "FoxitSerifItalic",
    SERIF_BOLD_ITALIC: "FoxitSerifBoldItalic",
    FIXED: "FoxitFixed",
    FIXED_BOLD: "FoxitFixedBold",
    FIXED_ITALIC: "FoxitFixedItalic",
    FIXED_BOLD_ITALIC: "FoxitFixedBoldItalic",
    SYMBOL: "FoxitSymbol",
    DINGBATS: "FoxitDingbats",
}

/// The descriptor flags (Table 121) a substitute is chosen by.
const FIXED_PITCH: i64 = 1;
const SERIF_FLAG: i64 = 1 << 1;
const ITALIC: i64 = 1 << 6;

/// The substitute of a font that embeds no program: `standard`, the standard font its name names (as
/// `crate::font::base14` canonicalizes it), else by `symbols`, else by its descriptor's `flags` and whether it is
/// `bold` or `italic`. The second value: whether it stands for a font other than a standard one (its glyphs are
/// stretched to the stated widths). None for a symbol font with no substitute.
pub(crate) fn substitute(
    standard: Option<&str>,
    symbols: Option<Symbols>,
    flags: i64,
    bold: bool,
    italic: bool,
) -> Option<(&'static Substitute, bool)> {
    let styled = |plain, b, i, bi| match (bold, italic) {
        (false, false) => plain,
        (true, false) => b,
        (false, true) => i,
        (true, true) => bi,
    };
    let sans = || styled(&SANS, &SANS_BOLD, &SANS_ITALIC, &SANS_BOLD_ITALIC);
    let serif = || styled(&SERIF, &SERIF_BOLD, &SERIF_ITALIC, &SERIF_BOLD_ITALIC);
    let fixed = || styled(&FIXED, &FIXED_BOLD, &FIXED_ITALIC, &FIXED_BOLD_ITALIC);
    if let Some(standard) = standard {
        let family = standard.split('-').next().unwrap_or(standard);
        return Some((
            match family {
                "Helvetica" => sans(),
                "Times" => serif(),
                "Courier" => fixed(),
                "Symbol" => &SYMBOL,
                _ => &DINGBATS,
            },
            false,
        ));
    }
    match symbols {
        Some(Symbols::Symbol) => return Some((&SYMBOL, false)),
        Some(Symbols::ZapfDingbats) => return Some((&DINGBATS, false)),
        Some(_) => return None,
        None => {}
    }
    let substitute = if flags & FIXED_PITCH != 0 {
        fixed()
    } else if flags & SERIF_FLAG != 0 {
        serif()
    } else {
        sans()
    };
    Some((substitute, true))
}

/// Every fallback program a font may be drawn with, by name: Noto Sans and Noto Serif CJK for each of Adobe's CJK
/// collections (Simplified and Traditional Chinese, Japanese, Korean), and Noto Sans Symbols 2, Noto Sans Symbols and
/// Noto Sans Math for the Microsoft symbol fonts' characters. None ships here: each is fetched by name, when a glyph
/// first needs it, through the file's [`Fetch`] (`document_lab.fonts` pins each name's download).
pub const FALLBACKS: [&str; 11] = [
    "noto-sans-sc",
    "noto-sans-tc",
    "noto-sans-jp",
    "noto-sans-kr",
    "noto-serif-sc",
    "noto-serif-tc",
    "noto-serif-jp",
    "noto-serif-kr",
    "noto-sans-symbols-2",
    "noto-sans-symbols",
    "noto-sans-math",
];

/// The fallbacks a Microsoft symbol font's characters are drawn from, in turn (together they hold every character
/// of Wingdings 1-3 and Webdings but U+2042 and U+2E3F).
const SYMBOL_CHAIN: [&str; 3] = ["noto-sans-symbols-2", "noto-sans-symbols", "noto-sans-math"];

/// How a composite font's fallback glyphs (a CJK font that embeds no program) are set across: a simple font's
/// substitute and fallback are always stretched across the widths the file states.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FallbackWidths {
    /// Each glyph at the fallback's own advance, as `MuPDF` sets its CJK substitute: a proportional Latin letter
    /// wider than the file's width overlaps the next.
    #[default]
    Own,
    /// Each glyph stretched across the width the file states for its `CID` (`/W`, else `/DW`), as a simple font's
    /// substitute is.
    Stated,
}

/// Fetches the fallback program a name of [`FALLBACKS`] names: its bytes, or why it cannot be had, in words.
pub type Fetch = dyn Fn(&str) -> std::result::Result<Vec<u8>, String>;

/// The fallback programs a file draws with: each fetched once, when a glyph first needs it, and kept, as is why it
/// could not be (a failure is not retried while the file is open).
#[derive(Clone, Default)]
pub(crate) struct Fallbacks(Rc<RefCell<Fetched>>);

#[derive(Default)]
struct Fetched {
    fetch: Option<Rc<Fetch>>,
    programs: HashMap<&'static str, Result<skrifa::Font, Rc<str>>>,
}

impl std::fmt::Debug for Fallbacks {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Fallbacks").finish_non_exhaustive()
    }
}

impl Fallbacks {
    /// Fetch through `fetch` from now on; what was fetched before is fetched again.
    pub(crate) fn set(&self, fetch: Rc<Fetch>) {
        let mut fetched = self.0.borrow_mut();
        fetched.fetch = Some(fetch);
        fetched.programs.clear();
    }

    /// The program `name` names, fetched and read once; why not, in words.
    fn program(&self, name: &'static str) -> Result<skrifa::Font, Rc<str>> {
        if let Some(program) = self.0.borrow().programs.get(name) {
            return program.clone();
        }
        let fetch = self.0.borrow().fetch.clone(); // not borrowed while it runs: it may take long, or read the file
        let program = match fetch {
            None => {
                Err(format!("{name} is not at hand: nothing fetches fallback fonts here").into())
            }
            Some(fetch) => fetch(name).map_err(Rc::from).and_then(|data| {
                skrifa::Font::new(data, 0)
                    .ok_or_else(|| format!("{name} was fetched but does not read").into())
            }),
        };
        self.0.borrow_mut().programs.insert(name, program.clone());
        program
    }
}

/// The fallbacks a font that embeds no program and has no substitute is drawn with, in turn: a composite font's by
/// the collection its `CID`s are of (`ordering`), in Noto Serif CJK when its descriptor flags it serif, else Noto
/// Sans CJK, as `MuPDF` chooses; a Microsoft symbol font's by its characters. None for any other font.
pub(crate) fn fallback(
    ordering: Option<&str>,
    symbols: Option<Symbols>,
    flags: i64,
) -> Option<&'static [&'static str]> {
    const fn one(name: &'static &'static str) -> &'static [&'static str] {
        std::slice::from_ref(name)
    }
    let serif = flags & SERIF_FLAG != 0;
    let at = |sans: usize| Some(one(&FALLBACKS[if serif { sans + 4 } else { sans }]));
    match (ordering, symbols) {
        (Some("GB1"), _) => at(0),
        (Some("CNS1"), _) => at(1),
        (Some("Japan1"), _) => at(2),
        (Some("Korea1" | "KR"), _) => at(3),
        (
            None,
            Some(
                Symbols::Wingdings | Symbols::Wingdings2 | Symbols::Wingdings3 | Symbols::Webdings,
            ),
        ) => Some(&SYMBOL_CHAIN),
        _ => None,
    }
}

/// Whether a font's descriptor or name says it is italic.
pub(crate) fn italic(flags: i64, italic_angle: Option<f64>, name: &str) -> bool {
    flags & ITALIC != 0
        || italic_angle.is_some_and(|a| a != 0.0)
        || ["Italic", "Oblique"].iter().any(|w| name.contains(w))
}

/// A font's glyph shapes: its source, the program read from it (once, when a shape is first asked) and each code's
/// shape (once).
#[derive(Clone)]
pub(crate) struct Shapes {
    pub(crate) source: Source,
    program: OnceCell<Result<skrifa::Font, Rc<str>>>,
    made: RefCell<HashMap<u32, Shape>>,
}

impl std::fmt::Debug for Shapes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Shapes")
            .field("source", &self.source)
            .finish_non_exhaustive()
    }
}

impl Shapes {
    pub(crate) fn new(source: Source) -> Self {
        Self {
            source,
            program: OnceCell::new(),
            made: RefCell::new(HashMap::new()),
        }
    }

    /// The program, read; why not, in words.
    fn program(&self, font: &Font) -> Result<&skrifa::Font, Rc<str>> {
        self.program
            .get_or_init(|| {
                let (data, what): (skrifa::font::Blob, String) = match &self.source {
                    Source::Embedded(data) => (data.to_vec().into(), "its embedded program".into()),
                    Source::Substitute { substitute, .. } => (
                        substitute.data.into(),
                        format!("its substitute {}", substitute.name),
                    ),
                    Source::Procedures { .. } => {
                        return Err("a Type 3 font has no program".into());
                    }
                    Source::Fallback { .. } => {
                        return Err("a font drawn from fallbacks has no one program".into());
                    }
                    Source::Unavailable(why) => return Err(Rc::clone(why)),
                };
                skrifa::Font::new(data, 0)
                    .ok_or_else(|| format!("font {}: {what} does not read", font.info.name).into())
            })
            .as_ref()
            .map_err(Rc::clone)
    }
}

impl Font {
    /// The shape of the glyph `code` selects.
    #[must_use]
    pub fn shape(&self, code: u32) -> Shape {
        if let Some(shape) = self.shapes.made.borrow().get(&code) {
            return shape.clone();
        }
        let shape = self.make_shape(code);
        self.shapes.made.borrow_mut().insert(code, shape.clone());
        shape
    }

    /// A Type 3 font's procedure for `code` (the `/CharProcs` entry its encoding names), and the resources it
    /// draws with when the font has its own; None for another font, or a code naming no procedure.
    #[must_use]
    pub fn procedure(&self, file: &crate::File, code: u32) -> Option<(Stream, Option<Dict>)> {
        let Source::Procedures {
            procedures,
            resources,
        } = &self.shapes.source
        else {
            return None;
        };
        let name = self.glyph_name(code)?;
        match file.get(procedures, name.as_bytes()).ok()?? {
            crate::Object::Stream(stream) => Some((stream, resources.clone())),
            _ => None,
        }
    }

    fn make_shape(&self, code: u32) -> Shape {
        match &self.shapes.source {
            Source::Procedures { .. } => return Shape::Procedure,
            Source::Fallback {
                fallbacks,
                chain,
                stretched,
            } => return self.fallback_shape(code, fallbacks, chain, *stretched),
            _ => {}
        }
        let program = match self.shapes.program(self) {
            Ok(program) => program,
            Err(why) => return Shape::Missing(why),
        };
        let stretched = matches!(
            self.shapes.source,
            Source::Substitute {
                stretched: true,
                ..
            }
        );
        self.drawn(program, self.glyph(program, code), code, stretched)
    }

    /// The shape of the glyph `code` selects drawn from fallbacks: the first program of `chain` holding a glyph for
    /// the character the code stands for (by its collection's text of its `CID`, a symbol font's table, its glyph
    /// name, its `ToUnicode`), fetched when first needed. Missing, by name, when a program of the chain cannot be
    /// had before one holds the character, when none does, or when the code stands for no one character; a space
    /// none holds draws nothing.
    fn fallback_shape(
        &self,
        code: u32,
        fallbacks: &Fallbacks,
        chain: &'static [&'static str],
        stretched: bool,
    ) -> Shape {
        let name = &self.info.name;
        let Some(c) = self.character(code) else {
            return Shape::Missing(
                format!("font {name} is not embedded: a code stands for no character its fallbacks could draw")
                    .into(),
            );
        };
        for &fallback in chain {
            let program = match fallbacks.program(fallback) {
                Ok(program) => program,
                Err(why) => {
                    return Shape::Missing(
                        format!("font {name} is not embedded and its fallback {why}").into(),
                    );
                }
            };
            if let Some(glyph) = program.charmap().map_unicode(c) {
                return self.drawn(&program, glyph, code, stretched);
            }
        }
        if c.is_whitespace() {
            return Shape::Outline(Rc::from([]));
        }
        Shape::Missing(
            format!(
                "font {name} is not embedded and a character it shows is in none of its fallbacks ({})",
                chain.join(", ")
            )
            .into(),
        )
    }

    /// The one character `code` stands for, for drawing it from a fallback: a composite font's by its collection's
    /// text of the code's `CID`, a symbol font's by its table, a simple font's by its glyph name, else by its
    /// `ToUnicode`.
    fn character(&self, code: u32) -> Option<char> {
        let one = |text: &str| {
            let mut chars = text.chars();
            match (chars.next(), chars.next()) {
                (Some(c), None) => Some(c),
                _ => None,
            }
        };
        let stated = || {
            self.to_unicode
                .as_ref()
                .and_then(|c| c.text(code))
                .and_then(one)
        };
        match &self.kind {
            Kind::Composite {
                cmap, collection, ..
            } => collection
                .and_then(|c| c.text(cmap.cid(code)))
                .as_deref()
                .and_then(one)
                .or_else(stated),
            Kind::Simple { .. } => u8::try_from(code)
                .ok()
                .and_then(|b| self.symbols.and_then(|s| s.character(b)))
                .or_else(|| self.glyph_name(code).as_deref().and_then(unicode_of))
                .or_else(stated),
        }
    }

    /// The outline of `glyph` in `program`, drawn for `code`: in text space, an em being the program's units per em;
    /// when `stretched`, across the width the file states for the code.
    fn drawn(&self, program: &skrifa::Font, glyph: GlyphId, code: u32, stretched: bool) -> Shape {
        let Some(outline) = program.outline_glyphs().get(glyph) else {
            return Shape::Outline(Rc::from([])); // a glyph id past the program's glyphs draws nothing
        };
        let mut pen = Pen::default();
        let Ok(drawn) = outline.draw(&DrawOptions::new(), DrawMemory::Scratch, &mut pen) else {
            return Shape::Missing(
                format!(
                    "font {}: glyph {} does not draw",
                    self.info.name,
                    glyph.to_u32()
                )
                .into(),
            );
        };
        let em = f64::from(program.units_per_em().max(1));
        let mut across = 1.0 / em;
        if stretched
            && let (Some(stated), Some(advance)) = (self.stated_width(code), drawn.h_advance)
            && stated > 0.0
            && advance > 0.0
        {
            across = stated / f64::from(advance); // the stated width over the substitute's, both per em
        }
        let scale = |p: Point| Point {
            x: p.x * across,
            y: p.y / em,
        };
        Shape::Outline(pen.segments.into_iter().map(|s| s.mapped(scale)).collect())
    }

    /// The glyph `code` selects in `program`.
    fn glyph(&self, program: &skrifa::Font, code: u32) -> GlyphId {
        match &self.kind {
            Kind::Composite { cmap, .. } => {
                let cid = cmap.cid(code);
                if let Some(map) = &self.cid_to_gid {
                    let gid = usize::try_from(cid).ok().and_then(|i| map.get(i));
                    return gid.map_or(GlyphId::NOTDEF, |&g| GlyphId::new(u32::from(g)));
                }
                if program.is_cid_keyed() {
                    return program.glyph_by_cid(cid).unwrap_or(GlyphId::NOTDEF);
                }
                GlyphId::new(cid)
            }
            Kind::Simple { .. } => {
                let name = self.glyph_name(code);
                let truetype = self.info.subtype == "TrueType"
                    && matches!(self.shapes.source, Source::Embedded(_));
                let found = if truetype {
                    truetype_glyph(program, code, name.as_deref(), self.symbolic())
                } else {
                    named_glyph(program, code, name.as_deref())
                };
                found.unwrap_or(GlyphId::NOTDEF)
            }
        }
    }
}

/// The outline of the character `c` in the font program `program` (any kind `skrifa` reads, its glyph reached by
/// its Unicode `cmap`) and its advance: in ems, y up, the origin on the baseline.
///
/// # Errors
///
/// Refuses a program that does not read, a character it holds no glyph for and a glyph that does not draw.
pub fn character(program: &[u8], c: char) -> crate::Result<(Vec<Segment>, f64)> {
    let font = skrifa::Font::new(program.to_vec(), 0)
        .ok_or_else(|| crate::Error::invalid("the font program does not read"))?;
    let missing = || crate::Error::invalid(format!("the font program has no glyph for {c:?}"));
    let glyph = font.charmap().map_unicode(c).ok_or_else(missing)?;
    let outline = font.outline_glyphs().get(glyph).ok_or_else(missing)?;
    let mut pen = Pen::default();
    let drawn = outline
        .draw(&DrawOptions::new(), DrawMemory::Scratch, &mut pen)
        .map_err(|e| crate::Error::invalid(format!("the glyph for {c:?} does not draw: {e}")))?;
    let em = f64::from(font.units_per_em().max(1));
    let outline = pen
        .segments
        .into_iter()
        .map(|s| {
            s.mapped(|p| Point {
                x: p.x / em,
                y: p.y / em,
            })
        })
        .collect();
    Ok((outline, drawn.h_advance.map_or(0.0, f64::from) / em))
}

/// The glyph a simple font's `code`, of glyph name `name`, selects in a Type 1 or CFF program (or a substitute): by
/// the name, by another name of its character, by the program's own encoding, by its Unicode mapping.
fn named_glyph(program: &skrifa::Font, code: u32, name: Option<&str>) -> Option<GlyphId> {
    let found = name.and_then(|n| program.glyph_by_name(n));
    let unicode = name.and_then(unicode_of);
    let alias = || {
        let c = unicode?;
        let mut buffer = [0u8; 64];
        (0..8).find_map(|n| {
            agl::char_to_nth_name(c, n, &mut buffer).and_then(|alias| program.glyph_by_name(alias))
        })
    };
    let charmap = program.charmap();
    let native = || {
        charmap
            .encodings()
            .filter(|e| e.sfnt_ids().is_none() && e.kind() != EncodingKind::Unicode)
            .find_map(|e| charmap.map_code(e, code))
    };
    found
        .or_else(alias)
        .or_else(|| name.is_none().then(native).flatten())
        .or_else(|| unicode.and_then(|c| charmap.map_unicode(c)))
        .or_else(native)
}

/// The glyph a simple font's `code`, of glyph name `name`, selects in a TrueType program (ISO 32000-2 9.6.5.4):
/// a named glyph through the Unicode (3,1) subtable by its name's character, else the Mac OS Roman (1,0) subtable by
/// its code there, else the `post` table's names; an unnamed one, or a symbolic font's, through the (3,0) subtable
/// by the code (as is, or in the U+F000, U+F100 and U+F200 rows), else (1,0) by the code, else (3,1) by the code.
fn truetype_glyph(
    program: &skrifa::Font,
    code: u32,
    name: Option<&str>,
    symbolic: bool,
) -> Option<GlyphId> {
    let charmap = program.charmap();
    let subtable = |platform: u16, encoding: u16| {
        charmap
            .encodings()
            .find(|e| e.sfnt_ids() == Some((platform, encoding)))
    };
    let (unicode, symbol, mac) = (
        subtable(3, 1).or(subtable(0, 3)),
        subtable(3, 0),
        subtable(1, 0),
    );
    let by_name = || {
        let name = name?;
        let c = unicode_of(name);
        let in_unicode = unicode
            .zip(c)
            .and_then(|(e, c)| charmap.map_code(e, u32::from(c)));
        let in_mac = || {
            let roman =
                c.and_then(|c| (0..=255u8).find(|&b| pdf_encoding::MACROMAN.get(b) == Some(c)))?;
            charmap.map_code(mac?, u32::from(roman))
        };
        in_unicode
            .or_else(in_mac)
            .or_else(|| program.glyph_by_name(name))
    };
    let by_code = || {
        let in_symbol = symbol.and_then(|e| {
            [code, 0xF000 + code, 0xF100 + code, 0xF200 + code]
                .into_iter()
                .find_map(|c| charmap.map_code(e, c))
        });
        in_symbol
            .or_else(|| mac.and_then(|e| charmap.map_code(e, code)))
            .or_else(|| unicode.and_then(|e| charmap.map_code(e, code)))
    };
    if symbolic && symbol.is_some() {
        by_code().or_else(by_name)
    } else {
        by_name().or_else(by_code)
    }
}

/// The one character a glyph name stands for, when it stands for one.
fn unicode_of(name: &str) -> Option<char> {
    let text = glyph_text(name)?;
    let mut chars = text.chars();
    match (chars.next(), chars.next()) {
        (Some(c), None) => Some(c),
        _ => None,
    }
}

/// Collects an outline as segments, a quadratic curve raised to a cubic.
#[derive(Default)]
struct Pen {
    segments: Vec<Segment>,
    at: Point,
}

impl PathSink for Pen {
    fn move_to(&mut self, x: f32, y: f32) {
        self.at = point(x, y);
        self.segments.push(Segment::MoveTo(self.at));
    }

    fn line_to(&mut self, x: f32, y: f32) {
        self.at = point(x, y);
        self.segments.push(Segment::LineTo(self.at));
    }

    fn quad_to(&mut self, cx0: f32, cy0: f32, x: f32, y: f32) {
        let (c, end) = (point(cx0, cy0), point(x, y));
        let third = |a: Point, b: Point| Point {
            x: a.x + (b.x - a.x) * 2.0 / 3.0,
            y: a.y + (b.y - a.y) * 2.0 / 3.0,
        };
        self.segments
            .push(Segment::CurveTo(third(self.at, c), third(end, c), end));
        self.at = end;
    }

    fn curve_to(&mut self, cx0: f32, cy0: f32, cx1: f32, cy1: f32, x: f32, y: f32) {
        self.at = point(x, y);
        self.segments
            .push(Segment::CurveTo(point(cx0, cy0), point(cx1, cy1), self.at));
    }

    fn close(&mut self) {
        self.segments.push(Segment::Close);
    }
}

fn point(x: f32, y: f32) -> Point {
    Point {
        x: f64::from(x),
        y: f64::from(y),
    }
}
