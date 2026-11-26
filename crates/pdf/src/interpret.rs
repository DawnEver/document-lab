//! The content-stream interpreter (ISO 32000-2 sections 8-9): what a page draws, as events in page space.
//!
//! The graphics state is tracked across `q`/`Q`, `cm`, colors, line style and Form `XObject`s; each painted path,
//! each image placed, each string shown and each shading painted (`sh`) is an event, its geometry already in page
//! space (points, y down). A fill or stroke with a pattern is a path event whose color names the pattern
//! ([`crate::color::Pattern`]); a pattern's own content (a tiling cell, a shading's colors) is not interpreted.
//!
//! The clip is tracked as the box it lies in -- the page's crop box, narrowed by each clipping path's bounds, each
//! Form's `/BBox` and, at `ET`, the union of the boxes of the glyphs shown in a clipping text mode (4-7) -- and a
//! mark wholly outside it is not seen: a path, image or shading gives no event, a glyph is left out of its text
//! event. A box holds the true clip, so a mark outside a curved clip (or between two clipping glyphs) but inside
//! its box is kept: never one lost, at worst one kept that a viewer hides. A shading's event is the box it may
//! paint: the clip's, within the shading's own `/BBox`, not the narrower area a non-extended gradient covers.
//!
//! Marked content (14.6) is tracked as a stack apart from the graphics state: `q`/`Q` neither saves nor ends it,
//! and a Form's sequences end with the Form (one left open is closed there, with a warning). What optional
//! content the default view hides ([`crate::optional`]) draws -- in a sequence marked `/OC`, or in a Form, image
//! or annotation whose own `/OC` is hidden -- is still an event, its [`Layer::hidden`] set: the reader reports
//! what the page holds and its caller chooses what to keep. An `/ActualText` (14.9.4) replaces the text of the
//! glyphs its sequence shows: the first glyph seen in it carries the replacement, every later one the empty text,
//! each keeping its own box; the outermost replacement governs a nested one, and one whose glyphs are all unseen is
//! lost with them.
//!
//! Annotations are drawn when asked, after the page's content: each one's normal appearance (`/AP /N`, its `/AS`
//! state's when it has states) mapped onto its `/Rect` (12.5.5), from the page's initial state, its events in the
//! annotation's [`Layer::annotation`]. A popup, an annotation flagged Hidden or `NoView`, and one with no
//! appearance draw nothing: no appearance is made up from an annotation's other entries.
//!
//! Not interpreted: a pattern's content and optional content usage ([`crate::optional`] says which). An operation
//! that cannot be read is skipped and named in [`Content::warnings`], never silently.

use std::collections::HashMap;
use std::rc::Rc;

use crate::color::{Color, Space};
use crate::content::{Operation, operations};
use crate::error::{Error, Result};
use crate::file::File;
use crate::font::{Font, FontInfo};
use crate::geometry::{Matrix, Point, Rect};
use crate::object::{Dict, Object, Stream, text_string};
use crate::optional::OptionalContent;
use crate::page::Page;

/// How deep Form `XObject`s may nest before one is refused as a loop.
const MAX_FORM_DEPTH: usize = 32;

/// One piece of a path, in page space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Segment {
    /// Start a subpath.
    MoveTo(Point),
    /// A straight line.
    LineTo(Point),
    /// A cubic Bezier curve: two control points, then its end.
    CurveTo(Point, Point, Point),
    /// Close the subpath back to its start.
    Close,
}

/// How a path is stroked.
#[derive(Debug, Clone, PartialEq)]
pub struct Stroke {
    /// The stroking color.
    pub color: Color,
    /// The line width in page space (points).
    pub width: f64,
    /// The dash array in page space (on, off, ...); empty for a solid line.
    pub dashes: Vec<f64>,
    /// Where in the dash array the line starts (page space).
    pub dash_phase: f64,
    /// The line cap style (0 butt, 1 round, 2 square).
    pub cap: i64,
    /// The line join style (0 miter, 1 round, 2 bevel).
    pub join: i64,
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
}

/// Where an event is drawn beyond the page's own visible content: in optional content the default view hides, in
/// an annotation's appearance. The default is the page's content, shown.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Layer {
    /// Optional content the default view hides draws it (see [`crate::optional`]).
    pub hidden: bool,
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
    /// An inline image's dictionary and data (still encoded); None for an image `XObject`.
    pub inline: Option<(Dict, Vec<u8>)>,
    /// The layer it is drawn in.
    pub layer: Layer,
}

/// One glyph of a text event.
#[derive(Debug, Clone, PartialEq)]
pub struct Glyph {
    /// Its character code in the font.
    pub code: u32,
    /// Its text: the font's for it, when the font says; inside an `/ActualText` sequence, the replacement on the
    /// sequence's first glyph seen and the empty text on every later one.
    pub text: Option<String>,
    /// Where its baseline starts, in page space.
    pub origin: Point,
    /// Where its baseline ends (its origin plus its width, before spacing), in page space.
    pub end: Point,
    /// The box from the font's descent to its ascent across the glyph's advance, in page space.
    pub bbox: Rect,
}

/// The glyphs one string shows, in one font, size and state.
#[derive(Debug, Clone, PartialEq)]
pub struct TextEvent {
    /// The font.
    pub font: Rc<FontInfo>,
    /// The font size in page space (the text matrix and the CTM applied).
    pub size: f64,
    /// The text rendering mode (`Tr`): 0 fill, 1 stroke, 3 invisible (an OCR text layer), 4-6 painted and added
    /// to the clip, 7 only added to the clip (invisible).
    pub render_mode: i64,
    /// The fill color.
    pub fill: Color,
    /// The stroke color.
    pub stroke: Color,
    /// Whether the font writes vertically (its glyphs are still placed along the baseline).
    pub vertical: bool,
    /// The glyphs, in showing order.
    pub glyphs: Vec<Glyph>,
    /// The layer it is drawn in.
    pub layer: Layer,
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

/// A shading painted with `sh`: a gradient across the clip.
#[derive(Debug, Clone, PartialEq)]
pub struct ShadingEvent {
    /// The box it may paint, in page space: the clip's, within the shading's `/BBox` when it has one.
    pub bbox: Rect,
    /// The layer it is drawn in.
    pub layer: Layer,
}

/// An open marked-content sequence (`BMC`/`BDC` ... `EMC`).
#[derive(Debug, Clone, Copy)]
struct Marked {
    /// Its optional content is hidden.
    hides: bool,
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
    stroke_space: Space,
    stroke_color: Color,
    fill_space: Space,
    fill_color: Color,
    line_width: f64,
    dashes: Vec<f64>,
    dash_phase: f64,
    cap: i64,
    join: i64,
    text: TextState,
    /// The box the clip lies in, page space.
    clip: Rect,
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
            clip,
            stroke_color: Space::Gray.initial(),
            stroke_space: Space::Gray,
            fill_color: Space::Gray.initial(),
            fill_space: Space::Gray,
            line_width: 1.0,
            dashes: Vec::new(),
            dash_phase: 0.0,
            cap: 0,
            join: 0,
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
    /// A clipping path is set (`W`, `W*`): the path ends by narrowing the clip, after it is painted.
    clipping: bool,
    /// The text matrix and the text line matrix (9.4.2), between `BT` and `ET`.
    text_matrix: Matrix,
    line_matrix: Matrix,
    /// The union of the boxes of the glyphs shown in a clipping mode in this text object: the clip at `ET`.
    text_clip: Option<Rect>,
    /// Fonts read, by the font dictionary's object number.
    fonts: HashMap<u32, Rc<Font>>,
    /// The default view of the file's optional content.
    optional: OptionalContent,
    /// The open marked-content sequences, outermost first.
    marked: Vec<Marked>,
    /// How many of them the running Form found open: its `EMC`s may not end those.
    marked_floor: usize,
    /// The outermost open `/ActualText`.
    actual: Option<ActualText>,
    /// How many of the `XObject`s and the annotation being drawn have an `/OC` entry that hides them.
    hiding_entries: usize,
    /// The `/Subtype` of the annotation whose appearance is being drawn.
    annotation: Option<String>,
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
        let mut interpreter = Interpreter {
            file: self,
            state: initial.clone(),
            stack: Vec::new(),
            path: Vec::new(),
            current: None,
            start: None,
            clipping: false,
            text_matrix: Matrix::IDENTITY,
            line_matrix: Matrix::IDENTITY,
            text_clip: None,
            fonts: HashMap::new(),
            optional: self.optional_content()?,
            marked: Vec::new(),
            marked_floor: 0,
            actual: None,
            hiding_entries: 0,
            annotation: None,
            out: Content::default(),
        };
        interpreter.run(&data, &page.resources, 0);
        interpreter.close_marked("the page");
        if annotations {
            interpreter.annotations(page, &initial);
        }
        Ok(interpreter.out)
    }
}

/// The last `count` operands, which must be numbers.
fn last(operands: &[Object], count: usize) -> Result<Vec<f64>> {
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
        let (ops, err) = operations(data);
        for op in &ops {
            if let Err(e) = self.operation(op, resources, depth) {
                self.out.warnings.push(format!(
                    "`{}` skipped: {e}",
                    String::from_utf8_lossy(&op.operator)
                ));
            }
        }
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
        let operator = op.operator.as_slice();
        let operands = &op.operands;
        if self.state_operation(operator, operands, resources)?
            || self.path_operation(operator, operands)?
            || self.paint_operation(operator)
            || self.color_operation(operator, operands, resources)?
            || self.text_operation(operator, operands, resources)?
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
                    self.image(
                        int(b"Width", b"W").unwrap_or(0),
                        int(b"Height", b"H").unwrap_or(0),
                        None,
                        Some((dict.clone(), data.clone())),
                    );
                }
            }
            b"W" | b"W*" => self.clipping = true,
            b"sh" => self.shading(operands, resources)?,
            b"BMC" => self.marked.push(Marked { hides: false }),
            b"BDC" => self.begin_marked(operands, resources),
            b"EMC" => self.end_marked()?,
            _ => {} // marked-content points and compatibility operators draw nothing
        }
        Ok(())
    }

    /// `BDC`: a sequence opens, hiding its content when its optional content is hidden, replacing its text when it
    /// has an `/ActualText`. Properties that cannot be read are a warning; the sequence opens all the same.
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
        let mut hides = false;
        let read = properties.and_then(|properties| {
            if tag == Some(b"OC") {
                hides = !self.optional.shows(self.file, &properties)?;
            }
            if self.actual.is_none()
                && let Object::Dict(dict) = self.file.resolve(&properties)?
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
        self.marked.push(Marked { hides });
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

    /// The layer what is painted now is drawn in.
    fn layer(&self) -> Layer {
        Layer {
            hidden: self.hiding_entries > 0 || self.marked.iter().any(|m| m.hides),
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
            let layer = self.layer();
            self.out
                .events
                .push(Event::Shading(ShadingEvent { bbox, layer }));
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
                let matrix = Matrix::from_numbers(&last(operands, 6)?)
                    .ok_or_else(|| Error::malformed("needs 6 numbers"))?;
                self.state.ctm = matrix.then(self.state.ctm);
            }
            b"w" => self.state.line_width = last(operands, 1)?[0],
            b"J" => self.state.cap = integer(operands)?,
            b"j" => self.state.join = integer(operands)?,
            b"d" => {
                let array = operands
                    .first()
                    .and_then(Object::as_array)
                    .ok_or_else(|| Error::malformed("needs a dash array"))?;
                self.state.dashes = array.iter().filter_map(Object::as_number).collect();
                self.state.dash_phase = operands.get(1).and_then(Object::as_number).unwrap_or(0.0);
            }
            b"gs" => self.ext_state(operands, resources)?,
            _ => return Ok(false),
        }
        Ok(true)
    }

    /// Path construction; whether `operator` is one.
    fn path_operation(&mut self, operator: &[u8], operands: &[Object]) -> Result<bool> {
        match operator {
            b"m" => {
                let v = last(operands, 2)?;
                let to = self.point(v[0], v[1]);
                self.path.push(Segment::MoveTo(to));
                (self.current, self.start) = (Some(to), Some(to));
            }
            b"l" => {
                let v = last(operands, 2)?;
                let to = self.point(v[0], v[1]);
                self.extend(Segment::LineTo(to), to);
            }
            b"c" => {
                let v = last(operands, 6)?;
                let (c1, c2, to) = (
                    self.point(v[0], v[1]),
                    self.point(v[2], v[3]),
                    self.point(v[4], v[5]),
                );
                self.extend(Segment::CurveTo(c1, c2, to), to);
            }
            b"v" => {
                let v = last(operands, 4)?;
                let c1 = self
                    .current
                    .ok_or_else(|| Error::malformed("no current point"))?;
                let (c2, to) = (self.point(v[0], v[1]), self.point(v[2], v[3]));
                self.extend(Segment::CurveTo(c1, c2, to), to);
            }
            b"y" => {
                let v = last(operands, 4)?;
                let (c1, to) = (self.point(v[0], v[1]), self.point(v[2], v[3]));
                self.extend(Segment::CurveTo(c1, to, to), to);
            }
            b"h" => {
                self.path.push(Segment::Close);
                self.current = self.start;
            }
            b"re" => {
                let v = last(operands, 4)?;
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
    ) -> Result<bool> {
        let text = &mut self.state.text;
        match operator {
            b"BT" => {
                (self.text_matrix, self.line_matrix) = (Matrix::IDENTITY, Matrix::IDENTITY);
                self.text_clip = None;
            }
            b"ET" => {
                if let Some(glyphs) = self.text_clip.take() {
                    self.state.clip = self.state.clip.intersect(glyphs);
                }
            }
            b"Tc" => text.char_spacing = last(operands, 1)?[0],
            b"Tw" => text.word_spacing = last(operands, 1)?[0],
            b"Tz" => text.scale = last(operands, 1)?[0] / 100.0,
            b"TL" => text.leading = last(operands, 1)?[0],
            b"Ts" => text.rise = last(operands, 1)?[0],
            b"Tr" => text.render_mode = integer(operands)?,
            b"Tf" => {
                text.size = last(operands, 1)?[0];
                let name = operands
                    .first()
                    .and_then(Object::as_name)
                    .ok_or_else(|| Error::malformed("needs a font name"))?;
                let font = self.load_font(name, resources)?;
                self.state.text.font = Some(font);
            }
            b"Td" => {
                let v = last(operands, 2)?;
                self.next_line(v[0], v[1]);
            }
            b"TD" => {
                let v = last(operands, 2)?;
                text.leading = -v[1];
                self.next_line(v[0], v[1]);
            }
            b"Tm" => {
                let m = Matrix::from_numbers(&last(operands, 6)?)
                    .ok_or_else(|| Error::malformed("needs 6 numbers"))?;
                (self.text_matrix, self.line_matrix) = (m, m);
            }
            b"T*" => {
                let leading = text.leading;
                self.next_line(0.0, -leading);
            }
            b"Tj" => self.show(string(operands)?),
            b"'" => {
                let leading = text.leading;
                self.next_line(0.0, -leading);
                self.show(string(operands)?);
            }
            b"\"" => {
                let v = last(&operands[..operands.len().saturating_sub(1)], 2)?;
                (text.word_spacing, text.char_spacing) = (v[0], v[1]);
                let leading = text.leading;
                self.next_line(0.0, -leading);
                self.show(string(operands)?);
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
                            let adjust = other.as_number().unwrap_or(0.0);
                            let text = &self.state.text;
                            self.advance(-adjust / 1000.0 * text.size * text.scale);
                        }
                    }
                }
                self.text_event(glyphs);
            }
            _ => return Ok(false),
        }
        Ok(true)
    }

    fn next_line(&mut self, tx: f64, ty: f64) {
        self.line_matrix = Matrix([1.0, 0.0, 0.0, 1.0, tx, ty]).then(self.line_matrix);
        self.text_matrix = self.line_matrix;
    }

    fn advance(&mut self, tx: f64) {
        self.text_matrix = Matrix([1.0, 0.0, 0.0, 1.0, tx, 0.0]).then(self.text_matrix);
    }

    fn load_font(&mut self, name: &[u8], resources: &Dict) -> Result<Rc<Font>> {
        let Some(Object::Dict(fonts)) = self.file.get(resources, b"Font")? else {
            return Err(Error::malformed("no /Font resources"));
        };
        let reference = fonts.get(name).ok_or_else(|| {
            Error::malformed(format!("no font /{}", String::from_utf8_lossy(name)))
        })?;
        let number = reference.as_reference().map(|r| r.num);
        if let Some(font) = number.and_then(|n| self.fonts.get(&n)) {
            return Ok(Rc::clone(font));
        }
        let Object::Dict(dict) = self.file.resolve(reference)? else {
            return Err(Error::malformed(format!(
                "font /{} is not a dictionary",
                String::from_utf8_lossy(name)
            )));
        };
        let font = Rc::new(self.file.font(&dict)?);
        if let Some(n) = number {
            self.fonts.insert(n, Rc::clone(&font));
        }
        Ok(font)
    }

    fn show(&mut self, bytes: &[u8]) {
        let glyphs = self.glyphs(bytes);
        self.text_event(glyphs);
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
            let (ascent, descent) = (
                font.ascent * font.matrix.0[3],
                font.descent * font.matrix.0[3],
            );
            let corners = [
                render.apply(Point { x: 0.0, y: descent }),
                render.apply(Point {
                    x: code.width,
                    y: descent,
                }),
                render.apply(Point {
                    x: code.width,
                    y: ascent,
                }),
                render.apply(Point { x: 0.0, y: ascent }),
            ];
            let space = if code.bytes == 1 && code.code == 32 {
                text.word_spacing
            } else {
                0.0
            };
            let advance = (code.width * text.size + text.char_spacing + space) * text.scale;
            out.push(Glyph {
                code: code.code,
                text: code.text,
                origin: render.apply(Point { x: 0.0, y: 0.0 }),
                end: render.apply(Point {
                    x: code.width,
                    y: 0.0,
                }),
                bbox: Rect::around(&corners).expect("four corners"),
            });
            self.advance(advance);
        }
        out
    }

    fn text_event(&mut self, mut glyphs: Vec<Glyph>) {
        if (4..=7).contains(&self.state.text.render_mode) {
            self.text_clip = glyphs
                .iter()
                .map(|g| g.bbox)
                .chain(self.text_clip)
                .reduce(Rect::union);
        }
        let Some(font) = &self.state.text.font else {
            return;
        };
        let clip = self.state.clip;
        glyphs.retain(|g| g.bbox.meets(clip));
        if glyphs.is_empty() {
            return;
        }
        if let Some(actual) = &mut self.actual {
            for glyph in &mut glyphs {
                glyph.text = Some(if actual.given {
                    String::new()
                } else {
                    actual.given = true;
                    actual.text.clone()
                });
            }
        }
        let text = &self.state.text;
        let m = self.text_matrix.then(self.state.ctm).0;
        let event = TextEvent {
            font: Rc::new(font.info.clone()),
            size: text.size * m[2].hypot(m[3]),
            render_mode: text.render_mode,
            fill: self.state.fill_color.clone(),
            stroke: self.state.stroke_color.clone(),
            vertical: font.vertical,
            glyphs,
            layer: self.layer(),
        };
        self.out.events.push(Event::Text(event));
    }

    /// Color spaces and colors; whether `operator` sets one.
    fn color_operation(
        &mut self,
        operator: &[u8],
        operands: &[Object],
        resources: &Dict,
    ) -> Result<bool> {
        let (space, color) = match operator {
            b"CS" | b"cs" => {
                let space = self
                    .file
                    .color_space(operands.first().unwrap_or(&Object::Null), resources)?;
                let color = space.initial();
                (space, color)
            }
            b"SC" | b"SCN" => {
                let space = self.state.stroke_space.clone();
                let color = self.color(&space, operands, resources)?;
                (space, color)
            }
            b"sc" | b"scn" => {
                let space = self.state.fill_space.clone();
                let color = self.color(&space, operands, resources)?;
                (space, color)
            }
            b"G" | b"g" | b"RG" | b"rg" | b"K" | b"k" => {
                let space = match operator {
                    b"G" | b"g" => Space::Gray,
                    b"RG" | b"rg" => Space::Rgb,
                    _ => Space::Cmyk,
                };
                let color = space.color(last(operands, space.components())?);
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

    /// The color `operands` set in `space`: in `Pattern`, the pattern the last operand names (after an uncolored
    /// pattern's tint).
    fn color(&self, space: &Space, operands: &[Object], resources: &Dict) -> Result<Color> {
        let mut color = space.color(numbers_in(operands));
        if *space == Space::Pattern {
            let name = operands
                .last()
                .and_then(Object::as_name)
                .ok_or_else(|| Error::malformed("needs a pattern name"))?;
            color.pattern = Some(self.file.pattern(name, resources)?);
        }
        Ok(color)
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

    /// The path, in `bounds`, has ended: when it clips, the clip narrows to them.
    fn ended(&mut self, bounds: Option<Rect>) {
        if std::mem::take(&mut self.clipping)
            && let Some(bounds) = bounds
        {
            self.state.clip = self.state.clip.intersect(bounds);
        }
        self.path.clear();
        (self.current, self.start) = (None, None);
    }

    fn paint(&mut self, stroke: bool, fill: Option<bool>) {
        let bounds = bounds(&self.path);
        if bounds.is_some_and(|b| b.meets(self.state.clip)) {
            let layer = self.layer();
            let state = &self.state;
            let scale = state.ctm.scale();
            let event = PathEvent {
                segments: std::mem::take(&mut self.path),
                stroke: stroke.then(|| Stroke {
                    color: state.stroke_color.clone(),
                    width: state.line_width * scale,
                    dashes: state.dashes.iter().map(|d| d * scale).collect(),
                    dash_phase: state.dash_phase * scale,
                    cap: state.cap,
                    join: state.join,
                }),
                fill: fill.map(|even_odd| Fill {
                    color: state.fill_color.clone(),
                    even_odd,
                }),
                layer,
            };
            self.out.events.push(Event::Path(event));
        }
        self.ended(bounds);
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
        Ok(())
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
                self.image(
                    int(b"Width"),
                    int(b"Height"),
                    reference.as_reference().map(|r| r.num),
                    None,
                );
                Ok(())
            }
            b"Form" => self.form(&stream, resources, depth),
            _ => Ok(()), // PostScript XObjects draw nothing a reader sees
        };
        self.hiding_entries -= hides;
        drawn
    }

    /// The Form `stream` drawn in the current state (8.10): through its `/Matrix`, clipped to its `/BBox`, with its
    /// own resources (else `resources`, its caller's); the state, the path and its marked content end with it.
    fn form(&mut self, stream: &Stream, resources: &Dict, depth: usize) -> Result<()> {
        if depth >= MAX_FORM_DEPTH {
            return Err(Error::malformed("Form XObjects nest too deep"));
        }
        let matrix = self.form_matrix(&stream.dict)?;
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
        if let Some(shown) = self.user_box(&stream.dict)? {
            self.state.clip = self.state.clip.intersect(shown);
        }
        let floor = std::mem::replace(&mut self.marked_floor, self.marked.len());
        self.run(&data, &own, depth + 1);
        self.close_marked("a Form XObject");
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
        for (i, annotation) in annotations.iter().enumerate() {
            if let Err(e) = self.annotation(annotation, initial) {
                self.out
                    .warnings
                    .push(format!("annotation {i} not drawn: {e}"));
            }
        }
        self.annotation = None;
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
        let stream = match self.file.get(&appearances, b"N")? {
            Some(Object::Stream(s)) => s,
            Some(Object::Dict(states)) => {
                let state = self.file.get(&annotation, b"AS")?;
                let state = state.as_ref().and_then(Object::as_name);
                match state
                    .and_then(|s| states.get(s))
                    .map(|s| self.file.resolve(s))
                {
                    Some(Ok(Object::Stream(s))) => s,
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
        let drawn = self.form(&stream, &Dict::new(), 0);
        self.hiding_entries -= hides;
        drawn
    }

    fn image(
        &mut self,
        width: i64,
        height: i64,
        object: Option<u32>,
        inline: Option<(Dict, Vec<u8>)>,
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
            layer: self.layer(),
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
