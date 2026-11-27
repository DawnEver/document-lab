//! The content-stream interpreter (ISO 32000-2 sections 8-9): what a page draws, as events in page space.
//!
//! The graphics state is tracked across `q`/`Q`, `cm`, colors, line style and Form `XObject`s; each painted path,
//! each image placed, each string shown and each shading painted (`sh`) is an event, its geometry already in page
//! space (points, y down). A fill or stroke with a pattern is a path event whose color names the pattern
//! ([`crate::color::Pattern`]: the pattern object, and its space mapped to page space through the default space of
//! the content stream it is painted in); an uncolored tiling pattern's tint is the color's, in the pattern space's
//! base. A pattern's own content is not interpreted with the page: a renderer draws a tiling cell through
//! [`File::cell_drawing`] and reads a shading through [`crate::shading`].
//!
//! What is painted in the colorant `None` ([`crate::color::Space::paints`]) marks nothing (8.6.6.4) and is still an
//! event, its [`Layer::inkless`] set, as hidden content is: a path filled or stroked in it -- a path whose fill marks
//! and whose stroke does not (or the reverse) is two events, the marking paint's and the inkless one's --, a stencil
//! mask painted in it, an image or a shading in a space of it alone. Text is not flagged: text shown in it is the
//! page's text as an invisible text layer's is (render mode 3, an OCR layer), which the reader reports as text.
//!
//! The clip is tracked as the box it lies in -- the page's crop box, narrowed by each clipping path's bounds, each
//! Form's `/BBox` and, at `ET`, the union of the boxes of the glyphs shown in a clipping text mode (4-7) -- and a
//! mark wholly outside it is not seen: a path, image or shading gives no event, a glyph is left out of its text
//! event. A box holds the true clip, so a mark outside a curved clip (or between two clipping glyphs) but inside
//! its box is kept: never one lost, at worst one kept that a viewer hides. A shading's event is the box it may
//! paint: the clip's, within the shading's own `/BBox`, not the narrower area a non-extended gradient covers.
//! Each event also carries the clip itself, for whoever draws it: the chain of clipping paths in force ([`Clip`]) --
//! each `W`/`W*` path, each Form's `/BBox`, each text object's clipping glyphs (by their outlines) -- and how it is
//! composited ([`Compositing`]: the constant alphas, the blend mode, the soft mask in force ([`SoftMask`]) and the
//! chain of transparency groups it is drawn in ([`Group`]), innermost first, as the clip is a chain). A text event
//! carries its font, each glyph's text rendering matrix and the stroke it is painted with: what drawing its glyphs'
//! shapes needs ([`crate::shape`]). [`File::drawing`] interprets a page for a renderer: as [`File::interpret`]
//! does, and each Type 3 glyph's procedure run in place after its text event, its marks events of their own. A soft
//! mask's group is drawn alone through [`File::form_drawing`], under the matrix in force where the mask was set.
//!
//! Marked content (14.6) is tracked as a stack apart from the graphics state: `q`/`Q` neither saves nor ends it,
//! and a Form's sequences end with the Form (one left open is closed there, with a warning). What optional
//! content the default view hides ([`crate::optional`]) draws -- in a sequence marked `/OC`, or in a Form, image
//! or annotation whose own `/OC` is hidden -- is still an event, its [`Layer::hidden`] set: the reader reports
//! what the page holds and its caller chooses what to keep. An `/ActualText` (14.9.4) replaces the text of the
//! glyphs its sequence shows: the first glyph seen in it carries the replacement ([`Glyph::actual_text`]), every
//! later one the empty text, each keeping its own box and the text its font gives it; the outermost replacement
//! governs a nested one, and one whose glyphs are all unseen is lost with them.
//!
//! Each event carries the marked content it is drawn in as a tagged PDF's structure tree refers to it ([`Marked`],
//! 14.7.5): the `/MCID` of the innermost sequence with one and the content stream holding that sequence -- the
//! page's, or a Form `XObject`'s -- so content a Form draws inside a page's sequence belongs to the page's; and
//! whether an `/Artifact` sequence (14.8.2.2) holds it.
//!
//! Annotations are drawn when asked, after the page's content: each one's normal appearance (`/AP /N`, its `/AS`
//! state's when it has states) mapped onto its `/Rect` (12.5.5), from the page's initial state, its events in the
//! annotation's [`Layer::annotation`]. A popup, an annotation flagged Hidden or `NoView`, and one with no
//! appearance draw nothing: no appearance is made up from an annotation's other entries. The interactive form's
//! default resources (`/AcroForm /DR`, 12.7.4.3: the resources form field appearance streams use) are an
//! appearance's resources when it has none, and give a font it names that its own resources lack. Its default
//! appearance string (`/DA`) is not applied: it is the recipe for making an appearance, not part of drawing one.
//!
//! Not interpreted with the page: a pattern's content and a soft mask's group (each drawn alone, above); optional
//! content usage ([`crate::optional`] says which). An operation
//! that cannot be read is skipped and named in [`Content::warnings`], never silently; a color whose space, profile
//! or tint transform does not read is set by its components alone, with no RGB, and named there too -- never left in
//! the space set before.
//!
//! The rendering intent (`ri`, a graphics state's `/RI`; relative colorimetric until one is set) is part of the
//! graphics state: a color in an `ICCBased` space converts at the intent in force when it is painted.

use std::rc::Rc;

use crate::color::{Color, Pattern, PatternKind, Space};
use crate::content::{Operation, operations};
use crate::error::{Error, Result};
use crate::file::File;
use crate::font::Font;
use crate::geometry::{Matrix, Point, Rect, Segment};
use crate::icc::Intent;
use crate::object::{Dict, Object, Stream, text_string};
use crate::optional::OptionalContent;
use crate::page::Page;
use crate::shape::Shape;

/// How deep Form `XObject`s may nest before one is refused as a loop.
const MAX_FORM_DEPTH: usize = 32;

/// Half an em in text space units of one font size (500 units of a composite font's 1000-unit glyph space): half a
/// vertical column's width across.
const HALF_EM: f64 = 0.5;

/// How a path is stroked.
#[derive(Debug, Clone, PartialEq)]
pub struct Stroke {
    /// The stroking color.
    pub color: Color,
    /// The line width in page space (points): the user space width scaled by [`Matrix::scale`] of `ctm`.
    pub width: f64,
    /// The dash array in page space (on, off, ...); empty for a solid line.
    pub dashes: Vec<f64>,
    /// Where in the dash array the line starts (page space).
    pub dash_phase: f64,
    /// The line cap style (0 butt, 1 round, 2 square).
    pub cap: i64,
    /// The line join style (0 miter, 1 round, 2 bevel).
    pub join: i64,
    /// The miter limit (`M`, `/ML`): the longest a miter join may be, in line widths, before it is beveled.
    pub miter_limit: f64,
    /// The matrix from user space to page space it was painted under (8.5.3.2: a line is stroked in user space,
    /// so under a matrix that scales unevenly its width varies with direction).
    pub ctm: Matrix,
}

/// A clipping path in force (8.5.4): what shows of a mark is what lies inside it and inside every clip it narrows.
#[derive(Debug, PartialEq)]
pub struct Clip {
    /// Its outline.
    pub outline: Outline,
    /// The clip in force when it was set, which it narrows; None: the page's alone.
    pub within: Option<Rc<Clip>>,
}

/// The outline of a clipping path.
#[derive(Debug, PartialEq)]
pub enum Outline {
    /// A path in page space (a clipping path's, a Form's `/BBox`), filled by the even-odd rule or the nonzero.
    Path {
        /// Its segments, page space.
        segments: Vec<Segment>,
        /// Even-odd (`W*`) rather than nonzero winding (`W`).
        even_odd: bool,
    },
    /// The glyphs shown in a clipping text mode (4-7) in one text object: the outlines of those whose shape is
    /// read, in page space and filled by the nonzero rule, and the boxes of those whose shape is not (a Type 3
    /// glyph, a font with no program to draw it from), which stand in for them.
    Glyphs {
        /// The outlines, page space.
        segments: Vec<Segment>,
        /// The boxes of the glyphs with no outline, page space.
        unshaped: Vec<Rect>,
    },
}

/// How a mark is composited onto what lies below it (11.3.5, 11.6.4): the graphics state's constant alphas, blend
/// mode and soft mask, and the transparency group it is drawn in.
#[derive(Debug, Clone, PartialEq)]
pub struct Compositing {
    /// The constant alpha (`/ca`) of filling: a fill, an image, a shading, text filled.
    pub fill_alpha: f64,
    /// The constant alpha (`/CA`) of stroking.
    pub stroke_alpha: f64,
    /// The blend mode (`/BM`).
    pub blend: Blend,
    /// The soft mask in force (`/SMask` a dictionary): what shows is further masked by it.
    pub soft_mask: Option<Rc<SoftMask>>,
    /// The innermost transparency group it is drawn in (a Form with a `/Group`, 11.6.6), whose result -- not each
    /// mark -- is composited onto what lies below as the group says; None: the page's. A group's own marks start
    /// from the initial alphas, blend mode and soft mask, as the standard has them.
    pub group: Option<Rc<Group>>,
}

impl Default for Compositing {
    fn default() -> Self {
        Self {
            fill_alpha: 1.0,
            stroke_alpha: 1.0,
            blend: Blend::Normal,
            soft_mask: None,
            group: None,
        }
    }
}

impl Compositing {
    /// Whether a group it is drawn in, at any depth, is composited otherwise than opaquely: with a constant alpha
    /// below 1, a blend mode other than `Normal` or a soft mask.
    #[must_use]
    pub fn in_composited_group(&self) -> bool {
        let mut group = self.group.as_deref();
        while let Some(g) = group {
            if g.compositing.translucent() {
                return true;
            }
            group = g.compositing.group.as_deref();
        }
        false
    }

    /// Whether what it composites shows otherwise than opaquely over what lies below: a constant alpha below 1, a
    /// blend mode other than `Normal` or a soft mask.
    #[must_use]
    pub fn translucent(&self) -> bool {
        self.fill_alpha < 1.0
            || self.stroke_alpha < 1.0
            || self.blend != Blend::Normal
            || self.soft_mask.is_some()
    }
}

/// A soft mask (11.6.5.2): a transparency group's drawing, read as each point's luminosity or alpha, through a
/// transfer function, masks what is painted while it is in force.
#[derive(Debug)]
pub struct SoftMask {
    /// Read as luminosity (`/S /Luminosity`), else as alpha (`/Alpha`).
    pub luminosity: bool,
    /// The group (`/G`): a Form `XObject`, drawn through [`File::form_drawing`].
    pub group: Rc<Stream>,
    /// Its object number, when it is an indirect object: what names one group set again and again.
    pub object: Option<u32>,
    /// The CTM where the mask was set: the group's space maps to page space through it, after its `/Matrix`.
    pub matrix: Matrix,
    /// The resources in force where it was set: the group's, when it has none of its own.
    pub resources: Dict,
    /// The backdrop's components (`/BC`) in the group's color space; empty when not stated (its initial color,
    /// black).
    pub backdrop: Vec<f64>,
    /// The transfer function (`/TR`) from the luminosity or alpha to the mask's value; None for the identity.
    pub transfer: Option<Object>,
}

impl PartialEq for SoftMask {
    /// The same mask set: one read is shared by every mark painted under it.
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self, other)
    }
}

impl SoftMask {
    /// The backdrop's RGB: `/BC` in the group's color space (`/Group /CS`), at `intent`; black when either is not
    /// stated.
    ///
    /// # Errors
    ///
    /// Refuses a color space that does not read, and a backdrop it gives no RGB.
    pub fn backdrop_rgb(&self, file: &File, intent: Intent) -> Result<[f64; 3]> {
        if self.backdrop.is_empty() {
            return Ok([0.0; 3]);
        }
        let space = match file.get(&self.group.dict, b"Group")? {
            Some(Object::Dict(group)) => match group.get(b"CS") {
                Some(cs) => Some(file.color_space(cs, &self.resources)?),
                None => None,
            },
            _ => None,
        };
        let Some(space) = space else {
            return Ok([0.0; 3]);
        };
        space
            .color(self.backdrop.clone(), intent)?
            .rgb
            .ok_or_else(|| Error::malformed("a soft mask's backdrop has no RGB"))
    }
}

/// A transparency group (11.6.6): a Form whose marks are composited together, then their result onto what lies
/// below.
#[derive(Debug)]
pub struct Group {
    /// Isolated (`/I true`): its marks are composited onto a transparent backdrop, not onto what lies below.
    pub isolated: bool,
    /// Knockout (`/K true`): each of its marks is composited onto the group's backdrop, not onto the marks before
    /// it.
    pub knockout: bool,
    /// How its result is composited: the alphas (the fill's), blend mode and soft mask in force where it was
    /// painted, and the group it is painted in.
    pub compositing: Compositing,
    /// The box it may paint, in page space: the clip's where it was painted, within its `/BBox`.
    pub bbox: Rect,
}

impl PartialEq for Group {
    /// The same group painted: one Form painted twice is two groups.
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self, other)
    }
}

/// A blend mode (11.3.5): how a mark's color mixes with the backdrop's.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Blend {
    /// `Normal` (and `Compatible`): the mark's color.
    #[default]
    Normal,
    /// `Multiply`.
    Multiply,
    /// `Screen`.
    Screen,
    /// `Overlay`.
    Overlay,
    /// `Darken`.
    Darken,
    /// `Lighten`.
    Lighten,
    /// `ColorDodge`.
    ColorDodge,
    /// `ColorBurn`.
    ColorBurn,
    /// `HardLight`.
    HardLight,
    /// `SoftLight`.
    SoftLight,
    /// `Difference`.
    Difference,
    /// `Exclusion`.
    Exclusion,
    /// `Hue`.
    Hue,
    /// `Saturation`.
    Saturation,
    /// `Color`.
    Color,
    /// `Luminosity`.
    Luminosity,
}

impl Blend {
    /// The blend mode `name` names; None for a name the standard does not define.
    #[must_use]
    pub fn named(name: &[u8]) -> Option<Self> {
        Some(match name {
            b"Normal" | b"Compatible" => Self::Normal,
            b"Multiply" => Self::Multiply,
            b"Screen" => Self::Screen,
            b"Overlay" => Self::Overlay,
            b"Darken" => Self::Darken,
            b"Lighten" => Self::Lighten,
            b"ColorDodge" => Self::ColorDodge,
            b"ColorBurn" => Self::ColorBurn,
            b"HardLight" => Self::HardLight,
            b"SoftLight" => Self::SoftLight,
            b"Difference" => Self::Difference,
            b"Exclusion" => Self::Exclusion,
            b"Hue" => Self::Hue,
            b"Saturation" => Self::Saturation,
            b"Color" => Self::Color,
            b"Luminosity" => Self::Luminosity,
            _ => return None,
        })
    }
}

/// How a path is filled.
#[derive(Debug, Clone, PartialEq)]
pub struct Fill {
    /// The filling color (none for a pattern).
    pub color: Color,
    /// Even-odd rather than nonzero winding.
    pub even_odd: bool,
}

/// A painted path.
#[derive(Debug, Clone, PartialEq)]
pub struct PathEvent {
    /// Its segments, in page space.
    pub segments: Vec<Segment>,
    /// The stroke, when stroked.
    pub stroke: Option<Stroke>,
    /// The fill, when filled.
    pub fill: Option<Fill>,
    /// The layer it is drawn in.
    pub layer: Layer,
    /// The marked content it is drawn in.
    pub marked: Marked,
    /// The clip in force; None: the page's alone.
    pub clip: Option<Rc<Clip>>,
    /// How it is composited.
    pub compositing: Compositing,
}

/// The marked content an event is drawn in (14.6), as a tagged PDF's structure tree refers to it (14.7.5).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Marked {
    /// The marked-content identifier (`/MCID`) of the innermost open sequence that has one; None outside them all.
    pub mcid: Option<i64>,
    /// The Form `XObject` (by object number) whose content stream holds that sequence; None for the page's content,
    /// and for an `XObject` written directly in a resource dictionary (it has no number to be referred to by).
    pub stream: Option<u32>,
    /// An `/Artifact` sequence holds it: pagination, layout or decoration, not the document's content (14.8.2.2).
    pub artifact: bool,
}

/// Where an event is drawn beyond the page's own visible content: in optional content the default view hides, in
/// the colorant `None`, in an annotation's appearance. The default is the page's content, shown.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Layer {
    /// Optional content the default view hides draws it (see [`crate::optional`]).
    pub hidden: bool,
    /// It is painted in the colorant `None`: it marks nothing, in any view (see the module).
    pub inkless: bool,
    /// The `/Subtype` of the annotation whose appearance draws it (`Widget`, `FreeText`, `Stamp`, ...); None for
    /// the page's content.
    pub annotation: Option<String>,
}

/// An image placed on the page.
#[derive(Debug, Clone, PartialEq)]
pub struct ImageEvent {
    /// The box the image covers, in page space.
    pub bbox: Rect,
    /// The matrix from the image's unit square to page space.
    pub matrix: Matrix,
    /// Its size in samples.
    pub width: i64,
    /// Its size in samples.
    pub height: i64,
    /// The image `XObject`'s object number; None for an inline image.
    pub object: Option<u32>,
    /// An inline image; None for an image `XObject`.
    pub inline: Option<Inline>,
    /// The fill color in force: what a stencil mask (`/ImageMask true`) paints.
    pub fill: Color,
    /// The layer it is drawn in.
    pub layer: Layer,
    /// The marked content it is drawn in.
    pub marked: Marked,
    /// The clip in force; None: the page's alone.
    pub clip: Option<Rc<Clip>>,
    /// How it is composited.
    pub compositing: Compositing,
}

/// An inline image (`BI` ... `ID` ... `EI`, 8.9.7).
#[derive(Debug, Clone, PartialEq)]
pub struct Inline {
    /// Its dictionary, its keys and names as written (abbreviated or not).
    pub dict: Dict,
    /// Its data, still encoded.
    pub data: Vec<u8>,
    /// The resources in force where it is placed: a color space it names is one of theirs.
    pub resources: Dict,
}

/// One glyph of a text event.
#[derive(Debug, Clone, PartialEq)]
pub struct Glyph {
    /// Its character code in the font.
    pub code: u32,
    /// Its text, as its font gives it; None when the font says nothing.
    pub text: Option<String>,
    /// Inside an `/ActualText` sequence, the text stated in place of the glyphs': the replacement on the sequence's
    /// first glyph seen, the empty text on every later one; None outside one.
    pub actual_text: Option<String>,
    /// Where it starts along its line, in page space: its origin on the baseline, or its vertical origin (the top
    /// of its place in a column) when its font writes vertically.
    pub origin: Point,
    /// Where it ends along its line (its origin plus its advance, before spacing), in page space.
    pub end: Point,
    /// Its box in page space: across its advance from the font's descent to its ascent, or when its font writes
    /// vertically, down its advance and an em across, centered on its origin.
    pub bbox: Rect,
    /// From its text space (an em is 1, y up: where its [`Shape`] lies) to page space: the text rendering matrix
    /// (9.4.4) -- font size, horizontal scaling, rise, text matrix, CTM -- its outline moved off the vertical origin by
    /// its position vector when its font writes vertically.
    pub matrix: Matrix,
}

/// The glyphs one string shows, in one font, size and state.
#[derive(Debug, Clone, PartialEq)]
pub struct TextEvent {
    /// The font.
    pub font: Rc<Font>,
    /// The font size in page space (the text matrix and the CTM applied).
    pub size: f64,
    /// The text rendering mode (`Tr`): 0 fill, 1 stroke, 3 invisible (an OCR text layer), 4-6 painted and added
    /// to the clip, 7 only added to the clip (invisible).
    pub render_mode: i64,
    /// The fill color.
    pub fill: Color,
    /// The stroke its glyphs are stroked with (render modes 1, 2, 5, 6): color, line width, dashes, caps, joins.
    pub stroke: Stroke,
    /// Whether the font writes vertically: its glyphs step down a column, each from its vertical origin.
    pub vertical: bool,
    /// The glyphs, in showing order.
    pub glyphs: Vec<Glyph>,
    /// The layer it is drawn in.
    pub layer: Layer,
    /// The marked content it is drawn in.
    pub marked: Marked,
    /// The clip in force; None: the page's alone.
    pub clip: Option<Rc<Clip>>,
    /// How it is composited.
    pub compositing: Compositing,
}

/// An event of a page, in drawing order.
#[derive(Debug, Clone, PartialEq)]
pub enum Event {
    /// A painted path.
    Path(PathEvent),
    /// A placed image.
    Image(ImageEvent),
    /// Text shown.
    Text(TextEvent),
    /// A shading painted over the clip (`sh`).
    Shading(ShadingEvent),
}

impl Event {
    /// How it is composited.
    #[must_use]
    pub fn compositing(&self) -> &Compositing {
        match self {
            Self::Path(e) => &e.compositing,
            Self::Image(e) => &e.compositing,
            Self::Text(e) => &e.compositing,
            Self::Shading(e) => &e.compositing,
        }
    }
}

/// A shading painted with `sh`: a gradient across the clip.
#[derive(Debug, Clone, PartialEq)]
pub struct ShadingEvent {
    /// The box it may paint, in page space: the clip's, within the shading's `/BBox` when it has one.
    pub bbox: Rect,
    /// The shading: its dictionary, or its stream (a mesh's), read by [`crate::shading`].
    pub shading: Rc<Object>,
    /// From the shading's space (the user space where it was painted) to page space: the CTM.
    pub matrix: Matrix,
    /// The rendering intent its colors convert at.
    pub intent: Intent,
    /// The layer it is drawn in.
    pub layer: Layer,
    /// The marked content it is drawn in.
    pub marked: Marked,
    /// The clip in force; None: the page's alone.
    pub clip: Option<Rc<Clip>>,
    /// How it is composited.
    pub compositing: Compositing,
}

/// An open marked-content sequence (`BMC`/`BDC` ... `EMC`).
#[derive(Debug, Clone, Copy, Default)]
struct Sequence {
    /// Its optional content is hidden.
    hides: bool,
    /// Its `/MCID`, and the Form `XObject` whose stream it is in (None: the page's).
    mcid: Option<(i64, Option<u32>)>,
    /// Its tag is `/Artifact`.
    artifact: bool,
}

/// The `/ActualText` replacing the text of the glyphs shown in the sequence at `depth` of the marked stack.
#[derive(Debug, Clone)]
struct ActualText {
    depth: usize,
    text: String,
    /// Whether a glyph carries it yet.
    given: bool,
}

/// What a page draws.
#[derive(Debug, Clone, Default)]
pub struct Content {
    /// Its events, in drawing order.
    pub events: Vec<Event>,
    /// Operations skipped and why, in words.
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone)]
struct State {
    ctm: Matrix,
    /// The matrix of the running content stream's default space (the page's, a Form's, a glyph procedure's, a
    /// tiling cell's) to page space: where a pattern's space lies (8.7.2).
    base: Matrix,
    stroke_space: Space,
    stroke_color: Color,
    fill_space: Space,
    fill_color: Color,
    line_width: f64,
    dashes: Vec<f64>,
    dash_phase: f64,
    cap: i64,
    join: i64,
    miter_limit: f64,
    text: TextState,
    /// The box the clip lies in, page space.
    clip: Rect,
    /// The clipping paths in force, innermost first; None: the page's alone.
    clips: Option<Rc<Clip>>,
    /// The rendering intent colors convert at.
    intent: Intent,
    compositing: Compositing,
}

/// The text state parameters (9.3), part of the graphics state.
#[derive(Debug, Clone)]
struct TextState {
    char_spacing: f64,
    word_spacing: f64,
    scale: f64,
    leading: f64,
    font: Option<Rc<Font>>,
    size: f64,
    render_mode: i64,
    rise: f64,
}

impl State {
    fn new(ctm: Matrix, clip: Rect) -> Self {
        Self {
            ctm,
            base: ctm,
            clip,
            stroke_color: Color::black(),
            stroke_space: Space::Gray,
            fill_color: Color::black(),
            fill_space: Space::Gray,
            line_width: 1.0,
            dashes: Vec::new(),
            dash_phase: 0.0,
            cap: 0,
            join: 0,
            miter_limit: 10.0,
            clips: None,
            intent: Intent::default(),
            compositing: Compositing::default(),
            text: TextState {
                char_spacing: 0.0,
                word_spacing: 0.0,
                scale: 1.0,
                leading: 0.0,
                font: None,
                size: 0.0,
                render_mode: 0,
                rise: 0.0,
            },
        }
    }
}

struct Interpreter<'a> {
    file: &'a File,
    state: State,
    stack: Vec<State>,
    path: Vec<Segment>,
    current: Option<Point>,
    start: Option<Point>,
    /// A clipping path is set (`W`, `W*`: whether by the even-odd rule): the path ends by narrowing the clip, after
    /// it is painted.
    clipping: Option<bool>,
    /// The text matrix and the text line matrix (9.4.2), between `BT` and `ET`.
    text_matrix: Matrix,
    line_matrix: Matrix,
    /// The glyphs shown in a clipping mode in this text object: the clip at `ET` (outlines, boxes of those without),
    /// and the box it lies in.
    text_clip: Option<(Vec<Segment>, Vec<Rect>, Rect)>,
    /// Whether each Type 3 glyph's procedure is run where the glyph is shown ([`File::drawing`]), and where: its
    /// text rendering matrix as the renderer places it.
    procedures: Option<&'a dyn Fn(Matrix) -> Matrix>,
    /// A Type 3 glyph's procedure is running.
    in_procedure: bool,
    /// ... after its `d1`: its color operators are ignored (9.6.4).
    uncolored: bool,
    /// The default view of the file's optional content.
    optional: OptionalContent,
    /// The open marked-content sequences, outermost first.
    marked: Vec<Sequence>,
    /// The Form `XObject` (by object number) whose content stream is running; None for the page's.
    stream: Option<u32>,
    /// How many of them the running Form found open: its `EMC`s may not end those.
    marked_floor: usize,
    /// The outermost open `/ActualText`.
    actual: Option<ActualText>,
    /// How many of the `XObject`s and the annotation being drawn have an `/OC` entry that hides them.
    hiding_entries: usize,
    /// The `/Subtype` of the annotation whose appearance is being drawn.
    annotation: Option<String>,
    /// The interactive form's default resources (`/AcroForm /DR`), read when annotations are drawn; empty when it
    /// has none.
    form_resources: Dict,
    out: Content,
}

impl File {
    /// What `page` draws: its content and, when `annotations` is asked, then its annotations' appearances.
    ///
    /// # Errors
    ///
    /// Refuses page contents that are not streams; an operation that cannot be read, or an annotation, is a warning
    /// instead.
    pub fn interpret(&self, page: &Page, annotations: bool) -> Result<Content> {
        self.interpreted(page, annotations, None)
    }

    /// What `page` draws, as a renderer needs it: [`File::interpret`]'s events, and after each text event in a Type
    /// 3 font that paints (modes other than 3 and 7), each of its glyphs' procedures run in place -- under its text
    /// rendering matrix as `place` places it (a renderer's own rounding of a glyph's origin) and the font's matrix,
    /// with the font's resources (else those in force) -- its marks events of their own (a procedure after `d1`
    /// paints in the fill color in force: its color operators are ignored).
    ///
    /// # Errors
    ///
    /// As [`File::interpret`].
    pub fn drawing(
        &self,
        page: &Page,
        annotations: bool,
        place: &dyn Fn(Matrix) -> Matrix,
    ) -> Result<Content> {
        self.interpreted(page, annotations, Some(place))
    }

    fn interpreted(
        &self,
        page: &Page,
        annotations: bool,
        procedures: Option<&dyn Fn(Matrix) -> Matrix>,
    ) -> Result<Content> {
        let mut data = Vec::new();
        match self.get(&page.dict, b"Contents")? {
            None => {}
            Some(Object::Stream(s)) => data = self.decode(&s)?.into_bytes()?,
            Some(Object::Array(parts)) => {
                for part in &parts {
                    if let Object::Stream(s) = self.resolve(part)? {
                        data.extend(self.decode(&s)?.into_bytes()?);
                        data.push(b'\n'); // a part may end mid-token; the parts join at a white-space
                    }
                }
            }
            Some(other) => {
                return Err(Error::malformed(format!(
                    "page {}: /Contents is a {}",
                    page.number,
                    other.kind()
                )));
            }
        }
        let (width, height) = page.size();
        let shown = Rect {
            x0: 0.0,
            y0: 0.0,
            x1: width,
            y1: height,
        };
        let initial = State::new(page.to_page_space(), shown);
        let mut interpreter = self.interpreter(initial.clone(), procedures)?;
        interpreter.run(&data, &page.resources, 0);
        interpreter.close_marked("the page");
        if annotations {
            interpreter.annotations(page, &initial);
        }
        Ok(interpreter.out)
    }

    /// An interpreter starting from `initial`, running Type 3 procedures when `procedures` places them.
    fn interpreter<'a>(
        &'a self,
        initial: State,
        procedures: Option<&'a dyn Fn(Matrix) -> Matrix>,
    ) -> Result<Interpreter<'a>> {
        Ok(Interpreter {
            file: self,
            state: initial,
            stack: Vec::new(),
            path: Vec::new(),
            current: None,
            start: None,
            clipping: None,
            text_matrix: Matrix::IDENTITY,
            line_matrix: Matrix::IDENTITY,
            text_clip: None,
            procedures,
            in_procedure: false,
            uncolored: false,
            optional: self.optional_content()?,
            marked: Vec::new(),
            stream: None,
            marked_floor: 0,
            actual: None,
            hiding_entries: 0,
            annotation: None,
            form_resources: Dict::new(),
            out: Content::default(),
        })
    }

    /// The Form `form` drawn alone, as a renderer draws a soft mask's group (11.6.5.2): as `Do` draws it --
    /// through its `/Matrix`, clipped to its `/BBox`, its marks a transparency group's when it is one -- under `ctm`
    /// from the initial graphics state, with its own resources (else `resources`), each Type 3 glyph's procedure run
    /// in place as [`File::drawing`] runs it, placed by `place`. Nothing outside `clip` (page space) is seen.
    ///
    /// # Errors
    ///
    /// Refuses a Form that does not decode; an operation that cannot be read is a warning instead.
    pub fn form_drawing(
        &self,
        form: &Stream,
        resources: &Dict,
        ctm: Matrix,
        clip: Rect,
        place: &dyn Fn(Matrix) -> Matrix,
    ) -> Result<Content> {
        let mut interpreter = self.interpreter(State::new(ctm, clip), Some(place))?;
        interpreter.form(form, None, resources, 0)?;
        Ok(interpreter.out)
    }

    /// A tiling pattern's cell drawn alone (8.7.3.1): its content stream under `ctm` (from pattern space), clipped
    /// to its `/BBox`, from the initial graphics state, with its own resources, each Type 3 glyph's procedure run in
    /// place as [`File::drawing`] runs it, placed by `place`. An uncolored pattern's cell (`tint` given) paints in the
    /// tint, its own color operators ignored. Nothing outside `clip` (page space) is seen.
    ///
    /// # Errors
    ///
    /// Refuses a pattern that is not a stream, and a cell that does not decode.
    pub fn cell_drawing(
        &self,
        pattern: &Object,
        ctm: Matrix,
        clip: Rect,
        tint: Option<&Color>,
        place: &dyn Fn(Matrix) -> Matrix,
    ) -> Result<Content> {
        let Object::Stream(cell) = pattern else {
            return Err(Error::malformed(format!(
                "a tiling pattern is a {}",
                pattern.kind()
            )));
        };
        let mut initial = State::new(ctm, clip);
        if let Some(tint) = tint {
            // The tint alone: the color naming the pattern paints its cell in it.
            let tint = Color {
                pattern: None,
                ..tint.clone()
            };
            (initial.fill_color, initial.stroke_color) = (tint.clone(), tint);
        }
        let mut interpreter = self.interpreter(initial, Some(place))?;
        interpreter.uncolored = tint.is_some();
        interpreter.form_under(cell, None, Matrix::IDENTITY, &Dict::new(), 0)?;
        Ok(interpreter.out)
    }
}

/// The last `N` operands, which must be numbers.
fn last<const N: usize>(operands: &[Object]) -> Result<[f64; N]> {
    let mut out = [0.0; N];
    let from = operands.len().checked_sub(N);
    for (i, slot) in out.iter_mut().enumerate() {
        *slot = from
            .and_then(|from| operands[from + i].as_number())
            .ok_or_else(|| Error::malformed(format!("needs {N} numbers")))?;
    }
    Ok(out)
}

/// The last `count` operands, which must be numbers: as many as a color space has components.
fn numbers(operands: &[Object], count: usize) -> Result<Vec<f64>> {
    operands
        .len()
        .checked_sub(count)
        .and_then(|from| {
            operands[from..]
                .iter()
                .map(Object::as_number)
                .collect::<Option<Vec<f64>>>()
        })
        .ok_or_else(|| Error::malformed(format!("needs {count} numbers")))
}

/// The last operand, a string.
fn string(operands: &[Object]) -> Result<&[u8]> {
    match operands.last() {
        Some(Object::String(s)) => Ok(s),
        _ => Err(Error::malformed("needs a string")),
    }
}

/// The last operand, an integer (a style code is whole).
fn integer(operands: &[Object]) -> Result<i64> {
    operands
        .last()
        .and_then(Object::as_integer)
        .ok_or_else(|| Error::malformed("needs an integer"))
}

impl Interpreter<'_> {
    fn run(&mut self, data: &[u8], resources: &Dict, depth: usize) {
        let err = operations(data, |op| {
            if let Err(e) = self.operation(op, resources, depth) {
                self.out.warnings.push(format!(
                    "`{}` skipped: {e}",
                    String::from_utf8_lossy(op.operator)
                ));
            }
        });
        if let Some(e) = err {
            self.out
                .warnings
                .push(format!("the content stream stops being readable: {e}"));
        }
    }

    fn point(&self, x: f64, y: f64) -> Point {
        self.state.ctm.apply(Point { x, y })
    }

    fn operation(&mut self, op: &Operation, resources: &Dict, depth: usize) -> Result<()> {
        let (operator, operands) = (op.operator, op.operands);
        if self.state_operation(operator, operands, resources)?
            || self.path_operation(operator, operands)?
            || self.paint_operation(operator)
            || self.color_operation(operator, operands, resources)?
            || self.text_operation(operator, operands, resources, depth)?
        {
            return Ok(());
        }
        match operator {
            b"Do" => self.x_object(operands, resources, depth)?,
            b"BI" => {
                if let Some((dict, data)) = &op.inline_image {
                    let int = |k: &[u8], short: &[u8]| {
                        dict.get(k)
                            .or_else(|| dict.get(short))
                            .and_then(Object::as_integer)
                    };
                    let inkless = self.image_inkless(dict, true, resources);
                    let inline = Inline {
                        dict: dict.clone(),
                        data: data.to_vec(),
                        resources: resources.clone(),
                    };
                    self.image(
                        int(b"Width", b"W").unwrap_or(0),
                        int(b"Height", b"H").unwrap_or(0),
                        (None, Some(inline)),
                        inkless,
                    );
                }
            }
            b"W" => self.clipping = Some(false),
            b"W*" => self.clipping = Some(true),
            b"sh" => self.shading(operands, resources)?,
            b"d1" if self.in_procedure => self.uncolored = true,
            b"BMC" => self.marked.push(Sequence {
                artifact: operands.first().and_then(Object::as_name) == Some(b"Artifact"),
                ..Sequence::default()
            }),
            b"BDC" => self.begin_marked(operands, resources),
            b"EMC" => self.end_marked()?,
            _ => {} // marked-content points and compatibility operators draw nothing
        }
        Ok(())
    }

    /// `BDC`: a sequence opens, hiding its content when its optional content is hidden, replacing its text when it
    /// has an `/ActualText`, identifying its content when it has an `/MCID`. Properties that cannot be read are a
    /// warning; the sequence opens all the same.
    fn begin_marked(&mut self, operands: &[Object], resources: &Dict) {
        let properties = match operands.get(1) {
            Some(Object::Name(name)) => match self.file.get(resources, b"Properties") {
                Ok(Some(Object::Dict(named))) => named.get(name).cloned().ok_or_else(|| {
                    Error::malformed(format!("no properties /{}", String::from_utf8_lossy(name)))
                }),
                Ok(_) => Err(Error::malformed("no /Properties resources")),
                Err(e) => Err(e),
            },
            Some(inline @ Object::Dict(_)) => Ok(inline.clone()),
            _ => Err(Error::malformed("needs a tag and properties")),
        };
        let tag = operands.first().and_then(Object::as_name);
        let mut sequence = Sequence {
            artifact: tag == Some(b"Artifact"),
            ..Sequence::default()
        };
        let read = properties.and_then(|properties| {
            if tag == Some(b"OC") {
                sequence.hides = !self.optional.shows(self.file, &properties)?;
            }
            let Object::Dict(dict) = self.file.resolve(&properties)? else {
                return Ok(());
            };
            if let Some(mcid) = self.file.get(&dict, b"MCID")?.and_then(|m| m.as_integer()) {
                sequence.mcid = Some((mcid, self.stream));
            }
            if self.actual.is_none()
                && let Some(Object::String(text)) = self.file.get(&dict, b"ActualText")?
            {
                self.actual = Some(ActualText {
                    depth: self.marked.len(),
                    text: text_string(&text),
                    given: false,
                });
            }
            Ok(())
        });
        if let Err(e) = read {
            self.out
                .warnings
                .push(format!("`BDC` read as plain marked content: {e}"));
        }
        self.marked.push(sequence);
    }

    /// `EMC`: the innermost open sequence ends.
    fn end_marked(&mut self) -> Result<()> {
        if self.marked.len() <= self.marked_floor {
            return Err(Error::malformed("no marked content is open"));
        }
        self.marked.pop();
        if self
            .actual
            .as_ref()
            .is_some_and(|a| a.depth >= self.marked.len())
        {
            self.actual = None;
        }
        Ok(())
    }

    /// The sequences `what` (the page, a Form) leaves open end with it, with a warning.
    fn close_marked(&mut self, what: &str) {
        let open = self.marked.len() - self.marked_floor;
        if open > 0 {
            self.out.warnings.push(format!(
                "{what} leaves {open} marked content sequence(s) open; they end with it"
            ));
            self.marked.truncate(self.marked_floor);
            if self
                .actual
                .as_ref()
                .is_some_and(|a| a.depth >= self.marked_floor)
            {
                self.actual = None;
            }
        }
    }

    /// The marked content what is painted now is drawn in.
    fn marked(&self) -> Marked {
        let (mcid, stream) = self
            .marked
            .iter()
            .rev()
            .find_map(|s| s.mcid)
            .map_or((None, None), |(mcid, stream)| (Some(mcid), stream));
        Marked {
            mcid,
            stream,
            artifact: self.marked.iter().any(|s| s.artifact),
        }
    }

    /// The layer what is painted now is drawn in.
    fn layer(&self) -> Layer {
        Layer {
            hidden: self.hiding_entries > 0 || self.marked.iter().any(|m| m.hides),
            inkless: false,
            annotation: self.annotation.clone(),
        }
    }

    /// Whether an `XObject` whose dictionary is `dict` shows by its `/OC`; one that cannot be read shows, with a
    /// warning.
    fn x_object_shows(&mut self, dict: &Dict) -> bool {
        let Some(oc) = dict.get(b"OC") else {
            return true;
        };
        self.optional.shows(self.file, oc).unwrap_or_else(|e| {
            self.out
                .warnings
                .push(format!("an XObject's /OC read as showing: {e}"));
            true
        })
    }

    /// `sh`: the shading `/name` paints the clip, within its `/BBox`.
    fn shading(&mut self, operands: &[Object], resources: &Dict) -> Result<()> {
        let name = operands
            .last()
            .and_then(Object::as_name)
            .ok_or_else(|| Error::malformed("needs a name"))?;
        let Some(Object::Dict(shadings)) = self.file.get(resources, b"Shading")? else {
            return Err(Error::malformed("no /Shading resources"));
        };
        let shading = shadings
            .get(name)
            .map(|s| self.file.resolve(s))
            .transpose()?
            .ok_or_else(|| {
                Error::malformed(format!("no shading /{}", String::from_utf8_lossy(name)))
            })?;
        let dict = shading.as_dict().ok_or_else(|| {
            Error::malformed(format!(
                "shading /{} is a {}",
                String::from_utf8_lossy(name),
                shading.kind()
            ))
        })?;
        let mut bbox = self.state.clip;
        if let Some(own) = self.user_box(dict)? {
            bbox = bbox.intersect(own);
        }
        if bbox.x0 <= bbox.x1 && bbox.y0 <= bbox.y1 {
            let inkless = dict
                .get(b"ColorSpace")
                .is_some_and(|space| self.file.colorant_none(space, resources));
            let layer = Layer {
                inkless,
                ..self.layer()
            };
            let matrix = self.state.ctm;
            let marked = self.marked();
            self.out.events.push(Event::Shading(ShadingEvent {
                bbox,
                shading: Rc::new(shading),
                matrix,
                intent: self.state.intent,
                layer,
                marked,
                clip: self.state.clips.clone(),
                compositing: self.state.compositing.clone(),
            }));
        }
        Ok(())
    }

    /// The box in page space of `dict`'s `/BBox` (a rectangle in user space), when it has one.
    fn user_box(&self, dict: &Dict) -> Result<Option<Rect>> {
        let Some([x0, y0, x1, y1]) = four(self.file.get(dict, b"BBox")?) else {
            return Ok(None);
        };
        let corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)].map(|(x, y)| self.point(x, y));
        Ok(Rect::around(&corners))
    }

    /// `q`, `Q`, `cm` and the line style; whether `operator` is one.
    fn state_operation(
        &mut self,
        operator: &[u8],
        operands: &[Object],
        resources: &Dict,
    ) -> Result<bool> {
        match operator {
            b"q" => self.stack.push(self.state.clone()),
            b"Q" => {
                if let Some(saved) = self.stack.pop() {
                    self.state = saved;
                }
            }
            b"cm" => {
                let matrix = Matrix::from_numbers(&last::<6>(operands)?)
                    .ok_or_else(|| Error::malformed("needs 6 numbers"))?;
                self.state.ctm = matrix.then(self.state.ctm);
            }
            b"w" => self.state.line_width = last::<1>(operands)?[0],
            b"J" => self.state.cap = integer(operands)?,
            b"j" => self.state.join = integer(operands)?,
            b"M" => self.state.miter_limit = last::<1>(operands)?[0],
            b"d" => {
                let array = operands
                    .first()
                    .and_then(Object::as_array)
                    .ok_or_else(|| Error::malformed("needs a dash array"))?;
                self.state.dashes = array.iter().filter_map(Object::as_number).collect();
                self.state.dash_phase = operands.get(1).and_then(Object::as_number).unwrap_or(0.0);
            }
            b"gs" => self.ext_state(operands, resources)?,
            b"ri" => {
                let name = operands
                    .last()
                    .and_then(Object::as_name)
                    .ok_or_else(|| Error::malformed("needs a name"))?;
                self.set_intent(name);
            }
            _ => return Ok(false),
        }
        Ok(true)
    }

    /// Path construction; whether `operator` is one.
    fn path_operation(&mut self, operator: &[u8], operands: &[Object]) -> Result<bool> {
        match operator {
            b"m" => {
                let v = last::<2>(operands)?;
                let to = self.point(v[0], v[1]);
                self.path.push(Segment::MoveTo(to));
                (self.current, self.start) = (Some(to), Some(to));
            }
            b"l" => {
                let v = last::<2>(operands)?;
                let to = self.point(v[0], v[1]);
                self.extend(Segment::LineTo(to), to);
            }
            b"c" => {
                let v = last::<6>(operands)?;
                let (c1, c2, to) = (
                    self.point(v[0], v[1]),
                    self.point(v[2], v[3]),
                    self.point(v[4], v[5]),
                );
                self.extend(Segment::CurveTo(c1, c2, to), to);
            }
            b"v" => {
                let v = last::<4>(operands)?;
                let c1 = self
                    .current
                    .ok_or_else(|| Error::malformed("no current point"))?;
                let (c2, to) = (self.point(v[0], v[1]), self.point(v[2], v[3]));
                self.extend(Segment::CurveTo(c1, c2, to), to);
            }
            b"y" => {
                let v = last::<4>(operands)?;
                let (c1, to) = (self.point(v[0], v[1]), self.point(v[2], v[3]));
                self.extend(Segment::CurveTo(c1, to, to), to);
            }
            b"h" => {
                self.path.push(Segment::Close);
                self.current = self.start;
            }
            b"re" => {
                let v = last::<4>(operands)?;
                let (left, bottom, right, top) = (v[0], v[1], v[0] + v[2], v[1] + v[3]);
                let corners = [
                    self.point(left, bottom),
                    self.point(right, bottom),
                    self.point(right, top),
                    self.point(left, top),
                ];
                self.path.push(Segment::MoveTo(corners[0]));
                for &corner in &corners[1..] {
                    self.path.push(Segment::LineTo(corner));
                }
                self.path.push(Segment::Close);
                (self.current, self.start) = (Some(corners[0]), Some(corners[0]));
            }
            _ => return Ok(false),
        }
        Ok(true)
    }

    /// Path painting; whether `operator` is one.
    fn paint_operation(&mut self, operator: &[u8]) -> bool {
        let (close, stroke, fill) = match operator {
            b"S" => (false, true, None),
            b"s" => (true, true, None),
            b"f" | b"F" => (false, false, Some(false)),
            b"f*" => (false, false, Some(true)),
            b"B" => (false, true, Some(false)),
            b"B*" => (false, true, Some(true)),
            b"b" => (true, true, Some(false)),
            b"b*" => (true, true, Some(true)),
            b"n" => {
                self.end_path();
                return true;
            }
            _ => return false,
        };
        if close {
            self.path.push(Segment::Close);
        }
        self.paint(stroke, fill);
        true
    }

    /// Text objects, state, positioning and showing; whether `operator` is one.
    fn text_operation(
        &mut self,
        operator: &[u8],
        operands: &[Object],
        resources: &Dict,
        depth: usize,
    ) -> Result<bool> {
        let text = &mut self.state.text;
        match operator {
            b"BT" => {
                (self.text_matrix, self.line_matrix) = (Matrix::IDENTITY, Matrix::IDENTITY);
                self.text_clip = None;
            }
            b"ET" => {
                if let Some((segments, unshaped, bounds)) = self.text_clip.take() {
                    self.state.clip = self.state.clip.intersect(bounds);
                    self.narrow(Outline::Glyphs { segments, unshaped });
                }
            }
            b"Tc" => text.char_spacing = last::<1>(operands)?[0],
            b"Tw" => text.word_spacing = last::<1>(operands)?[0],
            b"Tz" => text.scale = last::<1>(operands)?[0] / 100.0,
            b"TL" => text.leading = last::<1>(operands)?[0],
            b"Ts" => text.rise = last::<1>(operands)?[0],
            b"Tr" => text.render_mode = integer(operands)?,
            b"Tf" => {
                text.size = last::<1>(operands)?[0];
                let name = operands
                    .first()
                    .and_then(Object::as_name)
                    .ok_or_else(|| Error::malformed("needs a font name"))?;
                let font = self.load_font(name, resources)?;
                self.state.text.font = Some(font);
            }
            b"Td" => {
                let v = last::<2>(operands)?;
                self.next_line(v[0], v[1]);
            }
            b"TD" => {
                let v = last::<2>(operands)?;
                text.leading = -v[1];
                self.next_line(v[0], v[1]);
            }
            b"Tm" => {
                let m = Matrix::from_numbers(&last::<6>(operands)?)
                    .ok_or_else(|| Error::malformed("needs 6 numbers"))?;
                (self.text_matrix, self.line_matrix) = (m, m);
            }
            b"T*" => {
                let leading = text.leading;
                self.next_line(0.0, -leading);
            }
            b"Tj" => self.show(string(operands)?, resources, depth),
            b"'" => {
                let leading = text.leading;
                self.next_line(0.0, -leading);
                self.show(string(operands)?, resources, depth);
            }
            b"\"" => {
                let v = last::<2>(&operands[..operands.len().saturating_sub(1)])?;
                (text.word_spacing, text.char_spacing) = (v[0], v[1]);
                let leading = text.leading;
                self.next_line(0.0, -leading);
                self.show(string(operands)?, resources, depth);
            }
            b"TJ" => {
                let items = operands
                    .last()
                    .and_then(Object::as_array)
                    .ok_or_else(|| Error::malformed("needs an array"))?;
                let mut glyphs = Vec::new();
                for item in items {
                    match item {
                        Object::String(s) => glyphs.extend(self.glyphs(s)),
                        other => {
                            // A number is taken off the coordinate along the writing mode (9.4.3): a
                            // positive one moves the next glyph left, or down a column.
                            let adjust = other.as_number().unwrap_or(0.0) / 1000.0;
                            let text = &self.state.text;
                            if text.font.as_ref().is_some_and(|f| f.vertical()) {
                                self.advance(0.0, -adjust * text.size);
                            } else {
                                self.advance(-adjust * text.size * text.scale, 0.0);
                            }
                        }
                    }
                }
                self.text_event(glyphs, resources, depth);
            }
            _ => return Ok(false),
        }
        Ok(true)
    }

    fn next_line(&mut self, tx: f64, ty: f64) {
        self.line_matrix = Matrix([1.0, 0.0, 0.0, 1.0, tx, ty]).then(self.line_matrix);
        self.text_matrix = self.line_matrix;
    }

    fn advance(&mut self, tx: f64, ty: f64) {
        self.text_matrix = Matrix([1.0, 0.0, 0.0, 1.0, tx, ty]).then(self.text_matrix);
    }

    /// The font `resources` name `name` -- in an annotation's appearance, the form's default resources' when
    /// `resources` lack it.
    fn load_font(&mut self, name: &[u8], resources: &Dict) -> Result<Rc<Font>> {
        let mut reference = self.font_entry(resources, name)?;
        if reference.is_none() && self.annotation.is_some() {
            reference = self.font_entry(&self.form_resources, name)?;
        }
        let reference = reference.ok_or_else(|| {
            Error::malformed(format!("no font /{}", String::from_utf8_lossy(name)))
        })?;
        let number = reference.as_reference().map(|r| r.num);
        if let Some(font) = number.and_then(|n| self.file.fonts.borrow().get(&n).cloned()) {
            return Ok(font);
        }
        let Object::Dict(dict) = self.file.resolve(&reference)? else {
            return Err(Error::malformed(format!(
                "font /{} is not a dictionary",
                String::from_utf8_lossy(name)
            )));
        };
        let font = Rc::new(self.file.font(&dict)?);
        for warning in &font.warnings {
            self.out.warnings.push(format!(
                "font /{} ({}): {warning}",
                String::from_utf8_lossy(name),
                font.info.name
            ));
        }
        if let Some(n) = number {
            self.file.fonts.borrow_mut().insert(n, Rc::clone(&font));
        }
        Ok(font)
    }

    /// The entry `name` in the fonts of `resources`; None when they have no such font.
    fn font_entry(&self, resources: &Dict, name: &[u8]) -> Result<Option<Object>> {
        Ok(match self.file.get(resources, b"Font")? {
            Some(Object::Dict(fonts)) => fonts.get(name).cloned(),
            _ => None,
        })
    }

    fn show(&mut self, bytes: &[u8], resources: &Dict, depth: usize) {
        let glyphs = self.glyphs(bytes);
        self.text_event(glyphs, resources, depth);
    }

    /// The glyphs `bytes` shows, the text matrix advanced past each.
    fn glyphs(&mut self, bytes: &[u8]) -> Vec<Glyph> {
        let Some(font) = self.state.text.font.clone() else {
            self.out
                .warnings
                .push("text shown with no font set; skipped".to_string());
            return Vec::new();
        };
        let mut out = Vec::new();
        for code in font.codes(bytes) {
            let text = &self.state.text;
            let render = Matrix([text.size * text.scale, 0.0, 0.0, text.size, 0.0, text.rise])
                .then(self.text_matrix)
                .then(self.state.ctm);
            let space = if code.bytes == 1 && code.code == 32 {
                text.word_spacing
            } else {
                0.0
            };
            // Horizontally: the box spans the advance w0 along the baseline, from the font's descent to its ascent
            // across it. Vertically (9.7.4.3, 9.4.4): the origin is the vertical one; the box spans the advance w1y
            // down the column and an em across it, centered on the origin; Tc and Tw step down unscaled by Tz.
            // The outline's origin: the horizontal one, which a vertical font's position vector sets back from the
            // vertical origin the glyph is placed by (9.7.4.3).
            let (vx, vy) = code.position.unwrap_or((0.0, 0.0));
            let (end, corners, step) = match code.vertical_advance {
                None => {
                    let (ascent, descent) = (
                        font.ascent * font.matrix.0[3],
                        font.descent * font.matrix.0[3],
                    );
                    let corners = [
                        (0.0, descent),
                        (code.width, descent),
                        (code.width, ascent),
                        (0.0, ascent),
                    ];
                    let advance = (code.width * text.size + text.char_spacing + space) * text.scale;
                    ((code.width, 0.0), corners, (advance, 0.0))
                }
                Some(down) => {
                    let corners = [
                        (-HALF_EM, 0.0),
                        (HALF_EM, 0.0),
                        (HALF_EM, down),
                        (-HALF_EM, down),
                    ];
                    let advance = down * text.size + text.char_spacing + space;
                    ((0.0, down), corners, (0.0, advance))
                }
            };
            let at = |(x, y): (f64, f64)| render.apply(Point { x, y });
            out.push(Glyph {
                code: code.code,
                text: code.text,
                actual_text: None,
                origin: at((0.0, 0.0)),
                end: at(end),
                bbox: Rect::around(&corners.map(at)).expect("four corners"),
                matrix: Matrix([1.0, 0.0, 0.0, 1.0, -vx, -vy]).then(render),
            });
            self.advance(step.0, step.1);
        }
        out
    }

    fn text_event(&mut self, mut glyphs: Vec<Glyph>, resources: &Dict, depth: usize) {
        let Some(font) = self.state.text.font.clone() else {
            return;
        };
        if (4..=7).contains(&self.state.text.render_mode) {
            self.clip_glyphs(&font, &glyphs);
        }
        let clip = self.state.clip;
        glyphs.retain(|g| g.bbox.meets(clip));
        if glyphs.is_empty() {
            return;
        }
        if let Some(actual) = &mut self.actual {
            for glyph in &mut glyphs {
                glyph.actual_text = Some(if actual.given {
                    String::new()
                } else {
                    actual.given = true;
                    actual.text.clone()
                });
            }
        }
        let text = &self.state.text;
        let m = self.text_matrix.then(self.state.ctm).0;
        let paints = !matches!(text.render_mode, 3 | 7);
        let event = TextEvent {
            font: Rc::clone(&font),
            size: text.size * m[2].hypot(m[3]),
            render_mode: text.render_mode,
            fill: self.state.fill_color.clone(),
            stroke: self.stroke(),
            vertical: font.vertical(),
            glyphs,
            layer: self.layer(),
            marked: self.marked(),
            clip: self.state.clips.clone(),
            compositing: self.state.compositing.clone(),
        };
        let shown: Vec<(u32, Matrix)> = match self.procedures {
            Some(place) if paints && font.info.subtype == "Type3" => event
                .glyphs
                .iter()
                .map(|g| (g.code, place(g.matrix)))
                .collect(),
            _ => Vec::new(),
        };
        self.out.events.push(Event::Text(event));
        for (code, matrix) in shown {
            if let Err(e) = self.procedure(&font, code, matrix, resources, depth) {
                self.warn(
                    b"Tj",
                    &format!("a Type 3 glyph's procedure is not drawn: {e}"),
                );
            }
        }
    }

    /// The glyphs shown in a clipping mode join the text object's clip: each one's outline, in page space, else its
    /// box (a Type 3 glyph, a glyph whose shape is missing).
    fn clip_glyphs(&mut self, font: &Font, glyphs: &[Glyph]) {
        let Some(first) = glyphs.first() else {
            return; // an empty string adds nothing to the clip
        };
        let (segments, unshaped, bounds) = self
            .text_clip
            .get_or_insert_with(|| (Vec::new(), Vec::new(), first.bbox));
        for glyph in glyphs {
            *bounds = bounds.union(glyph.bbox);
            match font.shape(glyph.code) {
                Shape::Outline(outline) => {
                    let placed: Vec<Segment> = outline
                        .iter()
                        .map(|s| s.mapped(|p| glyph.matrix.apply(p)))
                        .collect();
                    if let Some(around) = self::bounds(&placed) {
                        *bounds = bounds.union(around);
                    }
                    segments.extend(placed);
                }
                Shape::Procedure | Shape::Missing(_) => unshaped.push(glyph.bbox),
            }
        }
    }

    /// The procedure of the Type 3 glyph `code` of `font` run, as a Form is (8.10), under the glyph's text rendering
    /// matrix `matrix` and the font's matrix, with the font's resources, else `resources`; the state, the path and
    /// the text object end with it.
    fn procedure(
        &mut self,
        font: &Font,
        code: u32,
        matrix: Matrix,
        resources: &Dict,
        depth: usize,
    ) -> Result<()> {
        let Some((stream, own)) = font.procedure(self.file, code) else {
            return Ok(()); // a code naming no procedure draws nothing
        };
        if depth >= MAX_FORM_DEPTH {
            return Err(Error::malformed("Type 3 glyphs and Forms nest too deep"));
        }
        let data = self.file.decode(&stream)?.into_bytes()?;
        let saved = (
            self.state.clone(),
            self.stack.len(),
            std::mem::take(&mut self.path),
            self.current,
            self.start,
            (self.text_matrix, self.line_matrix),
            self.text_clip.take(),
            (self.in_procedure, self.uncolored),
        );
        self.state.ctm = font.matrix.then(matrix);
        self.state.base = self.state.ctm;
        (self.in_procedure, self.uncolored) = (true, false);
        let floor = std::mem::replace(&mut self.marked_floor, self.marked.len());
        self.run(&data, own.as_ref().unwrap_or(resources), depth + 1);
        self.close_marked("a Type 3 glyph's procedure");
        self.marked_floor = floor;
        self.state = saved.0;
        self.stack.truncate(saved.1);
        (self.path, self.current, self.start) = (saved.2, saved.3, saved.4);
        (self.text_matrix, self.line_matrix) = saved.5;
        self.text_clip = saved.6;
        (self.in_procedure, self.uncolored) = saved.7;
        Ok(())
    }

    /// The stroke the graphics state paints with, in page space.
    fn stroke(&self) -> Stroke {
        let state = &self.state;
        let scale = state.ctm.scale();
        Stroke {
            color: state.stroke_color.clone(),
            width: state.line_width * scale,
            dashes: state.dashes.iter().map(|d| d * scale).collect(),
            dash_phase: state.dash_phase * scale,
            cap: state.cap,
            join: state.join,
            miter_limit: state.miter_limit,
            ctm: state.ctm,
        }
    }

    /// Color spaces and colors; whether `operator` sets one.
    fn color_operation(
        &mut self,
        operator: &[u8],
        operands: &[Object],
        resources: &Dict,
    ) -> Result<bool> {
        if self.uncolored
            && matches!(
                operator,
                b"CS"
                    | b"cs"
                    | b"SC"
                    | b"SCN"
                    | b"sc"
                    | b"scn"
                    | b"G"
                    | b"g"
                    | b"RG"
                    | b"rg"
                    | b"K"
                    | b"k"
            )
        {
            return Ok(true); // a Type 3 glyph after `d1` paints in the color in force
        }
        let (space, color) = match operator {
            b"CS" | b"cs" => {
                let value = operands.first().unwrap_or(&Object::Null);
                let space = match self.file.color_space(value, resources) {
                    Ok(space) => space,
                    Err(e) => {
                        // Its colors are known by their components alone until another space is set.
                        self.warn(operator, &format!("a color space that does not read: {e}"));
                        let named = value.as_name().or_else(|| {
                            value
                                .as_array()
                                .and_then(|a| a.first())
                                .and_then(Object::as_name)
                        });
                        Space::Other(named.unwrap_or(b"Unknown").to_vec(), 1)
                    }
                };
                if let Some(why) = space.unread() {
                    self.warn(operator, &format!("its colors have no RGB: {why}"));
                }
                let color = self.converted(operator, &space, space.initial_components());
                (space, color)
            }
            b"SC" | b"SCN" => {
                let space = self.state.stroke_space.clone();
                let color = self.color(operator, &space, operands, resources)?;
                (space, color)
            }
            b"sc" | b"scn" => {
                let space = self.state.fill_space.clone();
                let color = self.color(operator, &space, operands, resources)?;
                (space, color)
            }
            b"G" | b"g" | b"RG" | b"rg" | b"K" | b"k" => {
                let space = match operator {
                    b"G" | b"g" => Space::Gray,
                    b"RG" | b"rg" => Space::Rgb,
                    _ => Space::Cmyk,
                };
                let color =
                    self.converted(operator, &space, numbers(operands, space.components())?);
                (space, color)
            }
            _ => return Ok(false),
        };
        if operator[0].is_ascii_uppercase() {
            (self.state.stroke_space, self.state.stroke_color) = (space, color);
        } else {
            (self.state.fill_space, self.state.fill_color) = (space, color);
        }
        Ok(true)
    }

    fn warn(&mut self, operator: &[u8], message: &str) {
        self.out.warnings.push(format!(
            "`{}`: {message}",
            String::from_utf8_lossy(operator)
        ));
    }

    /// The color of `components` in `space`, at the intent in force; when the space gives them no RGB for a cause (a
    /// tint transform's error), the components alone, the cause named in a warning.
    fn converted(&mut self, operator: &[u8], space: &Space, components: Vec<f64>) -> Color {
        space
            .color(components.clone(), self.state.intent)
            .unwrap_or_else(|e| {
                self.warn(operator, &format!("its color has no RGB: {e}"));
                Color {
                    space: space.family(),
                    components,
                    rgb: None,
                    pattern: None,
                }
            })
    }

    /// The color `operands` set in `space`: in `Pattern`, the pattern the last operand names (after an uncolored
    /// pattern's tint).
    fn color(
        &mut self,
        operator: &[u8],
        space: &Space,
        operands: &[Object],
        resources: &Dict,
    ) -> Result<Color> {
        let pattern = if let Space::Pattern(_) = space {
            let name = operands
                .last()
                .and_then(Object::as_name)
                .ok_or_else(|| Error::malformed("needs a pattern name"))?;
            Some(
                self.file
                    .pattern(name, resources, self.state.base, self.state.intent)?,
            )
        } else {
            None
        };
        Ok(self.painted(operator, space, numbers_in(operands), pattern))
    }

    /// The color of `components` in `space` painting `pattern`: a pattern's has an RGB only when it is an uncolored
    /// tiling pattern's tint; another's components are the operands it was set by, if any.
    fn painted(
        &mut self,
        operator: &[u8],
        space: &Space,
        components: Vec<f64>,
        pattern: Option<Pattern>,
    ) -> Color {
        let tinted = matches!(
            pattern.as_ref().map(|p| p.kind),
            None | Some(PatternKind::Tiling { uncolored: true })
        );
        let mut color = if tinted {
            self.converted(operator, space, components)
        } else {
            Color {
                space: space.family(),
                components,
                rgb: None,
                pattern: None,
            }
        };
        // A pattern's colors convert at the intent in force, as a color's do.
        color.pattern = pattern.map(|p| Pattern {
            intent: self.state.intent,
            ..p
        });
        color
    }

    /// `ri`, a graphics state's `/RI`: the intent `name` names is in force, the colors set converted at it; a name the
    /// standard does not define leaves the default, relative colorimetric (8.6.5.8).
    fn set_intent(&mut self, name: &[u8]) {
        let intent = Intent::named(name).unwrap_or_default();
        if intent == self.state.intent {
            return;
        }
        self.state.intent = intent;
        for stroking in [true, false] {
            let (space, color) = if stroking {
                (&self.state.stroke_space, &self.state.stroke_color)
            } else {
                (&self.state.fill_space, &self.state.fill_color)
            };
            let (space, components, pattern) = (
                space.clone(),
                color.components.clone(),
                color.pattern.clone(),
            );
            let color = self.painted(b"ri", &space, components, pattern);
            if stroking {
                self.state.stroke_color = color;
            } else {
                self.state.fill_color = color;
            }
        }
    }

    fn extend(&mut self, segment: Segment, end: Point) {
        if self.current.is_none() {
            // A segment without a current point starts its own subpath (a lenient reading of an invalid stream).
            self.path.push(Segment::MoveTo(end));
            self.start = Some(end);
        } else {
            self.path.push(segment);
        }
        self.current = Some(end);
    }

    fn end_path(&mut self) {
        let bounds = bounds(&self.path);
        self.ended(bounds);
    }

    /// The path, in `bounds`, has ended: when it clips, the clip narrows to it (its box to `bounds`).
    fn ended(&mut self, bounds: Option<Rect>) {
        let segments = std::mem::take(&mut self.path);
        if let Some(even_odd) = self.clipping.take()
            && let Some(bounds) = bounds
        {
            self.state.clip = self.state.clip.intersect(bounds);
            self.narrow(Outline::Path { segments, even_odd });
        }
        (self.current, self.start) = (None, None);
    }

    /// The clip narrows to `outline`.
    fn narrow(&mut self, outline: Outline) {
        let within = self.state.clips.take();
        self.state.clips = Some(Rc::new(Clip { outline, within }));
    }

    fn paint(&mut self, stroke: bool, fill: Option<bool>) {
        let bounds = bounds(&self.path);
        if bounds.is_some_and(|b| b.meets(self.state.clip)) {
            // The paints that mark, then those in the colorant `None`, which mark nothing: an event each.
            let (strokes, fills) = (
                self.state.stroke_space.paints(),
                self.state.fill_space.paints(),
            );
            for inkless in [false, true] {
                let stroke = stroke && strokes != inkless;
                let fill = fill.filter(|_| fills != inkless);
                if stroke || fill.is_some() {
                    self.push_path(stroke, fill, inkless);
                }
            }
        }
        self.ended(bounds);
    }

    /// The path painted: stroked when `stroke`, filled by the even-odd rule or not when `fill` says, `inkless` when
    /// the paints are in the colorant `None`.
    fn push_path(&mut self, stroke: bool, fill: Option<bool>, inkless: bool) {
        let state = &self.state;
        let event = PathEvent {
            segments: self.path.clone(),
            stroke: stroke.then(|| self.stroke()),
            fill: fill.map(|even_odd| Fill {
                color: state.fill_color.clone(),
                even_odd,
            }),
            layer: Layer {
                inkless,
                ..self.layer()
            },
            marked: self.marked(),
            clip: state.clips.clone(),
            compositing: state.compositing.clone(),
        };
        self.out.events.push(Event::Path(event));
    }

    fn ext_state(&mut self, operands: &[Object], resources: &Dict) -> Result<()> {
        let name = operands
            .first()
            .and_then(Object::as_name)
            .ok_or_else(|| Error::malformed("needs a name"))?;
        let Some(Object::Dict(states)) = self.file.get(resources, b"ExtGState")? else {
            return Err(Error::malformed("no /ExtGState resources"));
        };
        let Some(Object::Dict(gs)) = states.get(name).map(|o| self.file.resolve(o)).transpose()?
        else {
            return Err(Error::malformed(format!(
                "no graphics state /{}",
                String::from_utf8_lossy(name)
            )));
        };
        if let Some(w) = self.file.get(&gs, b"LW")?.and_then(|o| o.as_number()) {
            self.state.line_width = w;
        }
        if let Some(c) = self.file.get(&gs, b"LC")?.and_then(|o| o.as_integer()) {
            self.state.cap = c;
        }
        if let Some(j) = self.file.get(&gs, b"LJ")?.and_then(|o| o.as_integer()) {
            self.state.join = j;
        }
        if let Some(m) = self.file.get(&gs, b"ML")?.and_then(|o| o.as_number()) {
            self.state.miter_limit = m;
        }
        let compositing = &mut self.state.compositing;
        if let Some(a) = self.file.get(&gs, b"CA")?.and_then(|o| o.as_number()) {
            compositing.stroke_alpha = a.clamp(0.0, 1.0);
        }
        if let Some(a) = self.file.get(&gs, b"ca")?.and_then(|o| o.as_number()) {
            compositing.fill_alpha = a.clamp(0.0, 1.0);
        }
        match self.file.get(&gs, b"BM")? {
            // An array names modes in order of preference: the first one known (11.6.3).
            Some(Object::Name(n)) => compositing.blend = Blend::named(&n).unwrap_or_default(),
            Some(Object::Array(names)) => {
                compositing.blend = names
                    .iter()
                    .find_map(|n| n.as_name().and_then(Blend::named))
                    .unwrap_or_default();
            }
            _ => {}
        }
        if let Some(Object::Name(intent)) = self.file.get(&gs, b"RI")? {
            self.set_intent(&intent);
        }
        if let Some(Object::Array(d)) = self.file.get(&gs, b"D")? {
            self.state.dashes = d
                .first()
                .and_then(Object::as_array)
                .unwrap_or(&[])
                .iter()
                .filter_map(Object::as_number)
                .collect();
            self.state.dash_phase = d.get(1).and_then(Object::as_number).unwrap_or(0.0);
        }
        // Last: a mask that does not read skips no other entry.
        match self.file.get(&gs, b"SMask")? {
            Some(Object::Dict(mask)) => {
                let soft_mask = self.soft_mask(&mask, resources)?;
                self.state.compositing.soft_mask = Some(Rc::new(soft_mask));
            }
            Some(Object::Name(n)) if n == b"None" => self.state.compositing.soft_mask = None,
            _ => {}
        }
        Ok(())
    }

    /// The soft mask the dictionary `mask` states (11.6.5.2), set where the CTM is the state's and `resources` are
    /// in force.
    fn soft_mask(&self, mask: &Dict, resources: &Dict) -> Result<SoftMask> {
        let luminosity = match self.file.get(mask, b"S")? {
            Some(Object::Name(s)) if s == b"Luminosity" => true,
            Some(Object::Name(s)) if s == b"Alpha" => false,
            _ => {
                return Err(Error::malformed(
                    "a soft mask's /S is neither Luminosity nor Alpha",
                ));
            }
        };
        let Some(Object::Stream(group)) = self.file.get(mask, b"G")? else {
            return Err(Error::malformed("a soft mask's /G is not a Form"));
        };
        let backdrop = match self.file.get(mask, b"BC")? {
            Some(Object::Array(bc)) => bc.iter().filter_map(Object::as_number).collect(),
            _ => Vec::new(),
        };
        let transfer = match self.file.get(mask, b"TR")? {
            Some(Object::Name(n)) if n == b"Identity" => None,
            other => other,
        };
        Ok(SoftMask {
            luminosity,
            group: Rc::new(group),
            object: mask.get(b"G").and_then(Object::as_reference).map(|r| r.num),
            matrix: self.state.ctm,
            resources: resources.clone(),
            backdrop,
            transfer,
        })
    }

    fn x_object(&mut self, operands: &[Object], resources: &Dict, depth: usize) -> Result<()> {
        let name = operands
            .first()
            .and_then(Object::as_name)
            .ok_or_else(|| Error::malformed("needs a name"))?;
        let Some(Object::Dict(objects)) = self.file.get(resources, b"XObject")? else {
            return Err(Error::malformed("no /XObject resources"));
        };
        let reference = objects.get(name).ok_or_else(|| {
            Error::malformed(format!("no XObject /{}", String::from_utf8_lossy(name)))
        })?;
        let Object::Stream(stream) = self.file.resolve(reference)? else {
            return Err(Error::malformed(format!(
                "XObject /{} is not a stream",
                String::from_utf8_lossy(name)
            )));
        };
        let subtype = stream
            .dict
            .get(b"Subtype")
            .and_then(Object::as_name)
            .unwrap_or(b"");
        let hides = usize::from(!self.x_object_shows(&stream.dict));
        self.hiding_entries += hides;
        let drawn = match subtype {
            b"Image" => {
                let int = |k: &[u8]| stream.dict.get(k).and_then(Object::as_integer).unwrap_or(0);
                let inkless = self.image_inkless(&stream.dict, false, resources);
                self.image(
                    int(b"Width"),
                    int(b"Height"),
                    (reference.as_reference().map(|r| r.num), None),
                    inkless,
                );
                Ok(())
            }
            b"Form" => self.form(
                &stream,
                reference.as_reference().map(|r| r.num),
                resources,
                depth,
            ),
            _ => Ok(()), // PostScript XObjects draw nothing a reader sees
        };
        self.hiding_entries -= hides;
        drawn
    }

    /// The Form `stream` (object `number`) drawn in the current state (8.10): through its `/Matrix`, clipped to its
    /// `/BBox`, with its own resources (else `resources`, its caller's); the state, the path and its marked content
    /// end with it.
    fn form(
        &mut self,
        stream: &Stream,
        number: Option<u32>,
        resources: &Dict,
        depth: usize,
    ) -> Result<()> {
        let matrix = self.form_matrix(&stream.dict)?;
        self.form_under(stream, number, matrix, resources, depth)
    }

    /// The Form (or tiling cell) `stream` drawn as [`Self::form`] draws it, through `matrix` rather than its
    /// `/Matrix`; when it is a transparency group (`/Group`), its marks are the group's.
    fn form_under(
        &mut self,
        stream: &Stream,
        number: Option<u32>,
        matrix: Matrix,
        resources: &Dict,
        depth: usize,
    ) -> Result<()> {
        if depth >= MAX_FORM_DEPTH {
            return Err(Error::malformed("Form XObjects nest too deep"));
        }
        let own = match self.file.get(&stream.dict, b"Resources")? {
            Some(Object::Dict(r)) => r,
            _ => resources.clone(),
        };
        let data = self.file.decode(stream)?.into_bytes()?;
        let saved = (
            self.state.clone(),
            self.stack.len(),
            std::mem::take(&mut self.path),
            self.current,
            self.start,
        );
        self.state.ctm = matrix.then(self.state.ctm);
        self.state.base = self.state.ctm;
        if let Some([x0, y0, x1, y1]) = four(self.file.get(&stream.dict, b"BBox")?) {
            let corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)].map(|(x, y)| self.point(x, y));
            let shown = Rect::around(&corners).expect("four corners");
            self.state.clip = self.state.clip.intersect(shown);
            let mut segments = vec![Segment::MoveTo(corners[0])];
            segments.extend(corners[1..].iter().map(|&c| Segment::LineTo(c)));
            segments.push(Segment::Close);
            self.narrow(Outline::Path {
                segments,
                even_odd: false,
            });
        }
        if let Some(Object::Dict(group)) = self.file.get(&stream.dict, b"Group")? {
            let flag = |key: &[u8]| -> Result<bool> {
                Ok(matches!(
                    self.file.get(&group, key)?,
                    Some(Object::Bool(true))
                ))
            };
            let group = Group {
                isolated: flag(b"I")?,
                knockout: flag(b"K")?,
                compositing: self.state.compositing.clone(),
                bbox: self.state.clip,
            };
            self.state.compositing = Compositing {
                group: Some(Rc::new(group)),
                ..Compositing::default()
            };
        }
        let floor = std::mem::replace(&mut self.marked_floor, self.marked.len());
        let caller = std::mem::replace(&mut self.stream, number);
        self.run(&data, &own, depth + 1);
        self.close_marked("a Form XObject");
        self.stream = caller;
        self.marked_floor = floor;
        self.state = saved.0;
        self.stack.truncate(saved.1);
        (self.path, self.current, self.start) = (saved.2, saved.3, saved.4);
        Ok(())
    }

    /// A Form's `/Matrix` (the identity when it has none).
    fn form_matrix(&self, dict: &Dict) -> Result<Matrix> {
        Ok(match self.file.get(dict, b"Matrix")? {
            Some(Object::Array(m)) => {
                Matrix::from_numbers(&m.iter().filter_map(Object::as_number).collect::<Vec<_>>())
                    .unwrap_or(Matrix::IDENTITY)
            }
            _ => Matrix::IDENTITY,
        })
    }

    /// Each annotation's appearance on `page`, drawn from the page's `initial` state; one that cannot be drawn is a
    /// warning.
    fn annotations(&mut self, page: &Page, initial: &State) {
        let annotations = match self.file.get(&page.dict, b"Annots") {
            Ok(Some(Object::Array(a))) => a,
            Ok(_) => return,
            Err(e) => {
                self.out.warnings.push(format!("/Annots unreadable: {e}"));
                return;
            }
        };
        match self.form_resources() {
            Ok(resources) => self.form_resources = resources,
            Err(e) => self.out.warnings.push(format!(
                "the form's default resources (/AcroForm /DR) unreadable: {e}"
            )),
        }
        for (i, annotation) in annotations.iter().enumerate() {
            if let Err(e) = self.annotation(annotation, initial) {
                self.out
                    .warnings
                    .push(format!("annotation {i} not drawn: {e}"));
            }
        }
        self.annotation = None;
    }

    /// The interactive form's default resources (`/AcroForm /DR`); empty when the file has none.
    fn form_resources(&self) -> Result<Dict> {
        let Some(Object::Dict(form)) = self.file.get(&self.file.catalog()?, b"AcroForm")? else {
            return Ok(Dict::new());
        };
        Ok(match self.file.get(&form, b"DR")? {
            Some(Object::Dict(resources)) => resources,
            _ => Dict::new(),
        })
    }

    /// The annotation `value`'s normal appearance (12.5.5), unless it is a popup or flagged hidden or not to view:
    /// its `/AP /N` -- the `/AS` state's, when it has states -- mapped from its `/BBox` (through its `/Matrix`) onto
    /// the annotation's `/Rect`, from the page's `initial` state.
    fn annotation(&mut self, value: &Object, initial: &State) -> Result<()> {
        /// `/F` bits: Hidden (2) and `NoView` (6).
        const NOT_SHOWN: i64 = 1 << 1 | 1 << 5;
        let Object::Dict(annotation) = self.file.resolve(value)? else {
            return Err(Error::malformed("not a dictionary"));
        };
        let subtype = self.file.get(&annotation, b"Subtype")?;
        let subtype = subtype.as_ref().and_then(Object::as_name).unwrap_or(b"");
        let flags = self
            .file
            .get(&annotation, b"F")?
            .and_then(|f| f.as_integer())
            .unwrap_or(0);
        if subtype == b"Popup" || flags & NOT_SHOWN != 0 {
            return Ok(());
        }
        let Some(Object::Dict(appearances)) = self.file.get(&annotation, b"AP")? else {
            return Ok(()); // no appearance: nothing drawn
        };
        let normal = appearances.get(b"N");
        let (stream, number) = match self.file.get(&appearances, b"N")? {
            Some(Object::Stream(s)) => (s, normal.and_then(Object::as_reference)),
            Some(Object::Dict(states)) => {
                let state = self.file.get(&annotation, b"AS")?;
                let state = state.as_ref().and_then(Object::as_name);
                let chosen = state.and_then(|s| states.get(s));
                match chosen.map(|s| self.file.resolve(s)) {
                    Some(Ok(Object::Stream(s))) => (s, chosen.and_then(Object::as_reference)),
                    Some(Err(e)) => return Err(e),
                    _ => return Ok(()), // no appearance for its state
                }
            }
            _ => return Ok(()),
        };
        let rect = four(self.file.get(&annotation, b"Rect")?)
            .ok_or_else(|| Error::malformed("its /Rect is not four numbers"))?;
        let rect = Rect::spanning(
            Point {
                x: rect[0],
                y: rect[1],
            },
            Point {
                x: rect[2],
                y: rect[3],
            },
        );
        let matrix = self.form_matrix(&stream.dict)?;
        let [b0, b1, b2, b3] = four(self.file.get(&stream.dict, b"BBox")?)
            .ok_or_else(|| Error::malformed("its appearance's /BBox is not four numbers"))?;
        let corners =
            [(b0, b1), (b2, b1), (b2, b3), (b0, b3)].map(|(x, y)| matrix.apply(Point { x, y }));
        let shown = Rect::around(&corners).expect("four corners");
        if shown.width() <= 0.0 || shown.height() <= 0.0 {
            return Err(Error::malformed("its appearance's /BBox is empty"));
        }
        let (sx, sy) = (rect.width() / shown.width(), rect.height() / shown.height());
        let onto = Matrix([
            sx,
            0.0,
            0.0,
            sy,
            rect.x0 - shown.x0 * sx,
            rect.y0 - shown.y0 * sy,
        ]);
        let hides = match annotation.get(b"OC") {
            Some(oc) => usize::from(!self.optional.shows(self.file, oc)?),
            None => 0,
        } + usize::from(!self.x_object_shows(&stream.dict));
        self.state = initial.clone();
        self.state.ctm = onto.then(initial.ctm);
        self.stack.clear();
        self.annotation = Some(String::from_utf8_lossy(subtype).into_owned());
        self.hiding_entries += hides;
        let defaults = self.form_resources.clone();
        let drawn = self.form(&stream, number.map(|r| r.num), &defaults, 0);
        self.hiding_entries -= hides;
        drawn
    }

    /// Whether the image whose dictionary is `dict` (an inline image's when `inline`, its keys and names
    /// abbreviated) marks nothing: a stencil mask in a fill color of the colorant `None`, or samples in a space of it
    /// alone.
    fn image_inkless(&self, dict: &Dict, inline: bool, resources: &Dict) -> bool {
        let key = |long: &'static [u8], short: &'static [u8]| {
            dict.get(long)
                .or_else(|| if inline { dict.get(short) } else { None })
        };
        let mask = key(b"ImageMask", b"IM").and_then(|m| self.file.resolve(m).ok());
        if matches!(mask, Some(Object::Bool(true))) {
            return !self.state.fill_space.paints();
        }
        key(b"ColorSpace", b"CS").is_some_and(|space| self.file.colorant_none(space, resources))
    }

    /// An image placed: `width` x `height` samples, the image `XObject`'s number or the inline image's dictionary and
    /// data, `inkless` when it marks nothing.
    fn image(
        &mut self,
        width: i64,
        height: i64,
        (object, inline): (Option<u32>, Option<Inline>),
        inkless: bool,
    ) {
        let m = self.state.ctm;
        let corners = [
            m.apply(Point { x: 0.0, y: 0.0 }),
            m.apply(Point { x: 1.0, y: 0.0 }),
            m.apply(Point { x: 1.0, y: 1.0 }),
            m.apply(Point { x: 0.0, y: 1.0 }),
        ];
        let bbox = Rect::around(&corners).expect("four corners");
        if !bbox.meets(self.state.clip) {
            return;
        }
        self.out.events.push(Event::Image(ImageEvent {
            bbox,
            matrix: m,
            width,
            height,
            object,
            inline,
            fill: self.state.fill_color.clone(),
            layer: Layer {
                inkless,
                ..self.layer()
            },
            marked: self.marked(),
            clip: self.state.clips.clone(),
            compositing: self.state.compositing.clone(),
        }));
    }
}

/// The box a path's points lie in -- a curve's control points too, so it holds the curve -- or None when empty.
fn bounds(path: &[Segment]) -> Option<Rect> {
    let points: Vec<Point> = path
        .iter()
        .flat_map(|s| match *s {
            Segment::MoveTo(p) | Segment::LineTo(p) => vec![p],
            Segment::CurveTo(a, b, p) => vec![a, b, p],
            Segment::Close => vec![],
        })
        .collect();
    Rect::around(&points)
}

fn numbers_in(operands: &[Object]) -> Vec<f64> {
    operands.iter().filter_map(Object::as_number).collect()
}

/// The four numbers `value` is an array of (a rectangle), if it is one.
fn four(value: Option<Object>) -> Option<[f64; 4]> {
    let numbers: Vec<f64> = value?
        .as_array()?
        .iter()
        .filter_map(Object::as_number)
        .collect();
    numbers.try_into().ok()
}
