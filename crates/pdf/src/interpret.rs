//! The content-stream interpreter (ISO 32000-2 sections 8-9): what a page draws, as events in page space.
//!
//! The graphics state is tracked across `q`/`Q`, `cm`, colors, line style and Form `XObject`s; each painted path
//! and each image placed is an event, its geometry already in page space (points, y down). Clipping and
//! shadings are not painted marks and give no event. An operation that cannot be read is skipped and named
//! in [`Content::warnings`], never silently.

use std::collections::HashMap;
use std::rc::Rc;

use crate::color::{Color, Space};
use crate::content::{Operation, operations};
use crate::error::{Error, Result};
use crate::file::File;
use crate::font::{Font, FontInfo};
use crate::geometry::{Matrix, Point, Rect};
use crate::object::{Dict, Object};
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
}

/// One glyph of a text event.
#[derive(Debug, Clone, PartialEq)]
pub struct Glyph {
    /// Its character code in the font.
    pub code: u32,
    /// Its text, when the font says what it is.
    pub text: Option<String>,
    /// Where its baseline starts, in page space.
    pub origin: Point,
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
    /// The text rendering mode (`Tr`): 0 fill, 1 stroke, 3 invisible (an OCR text layer), 7 clip, ...
    pub render_mode: i64,
    /// The fill color.
    pub fill: Color,
    /// The stroke color.
    pub stroke: Color,
    /// Whether the font writes vertically (its glyphs are still placed along the baseline).
    pub vertical: bool,
    /// The glyphs, in showing order.
    pub glyphs: Vec<Glyph>,
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
    fn new(ctm: Matrix) -> Self {
        Self {
            ctm,
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
    /// The text matrix and the text line matrix (9.4.2), between `BT` and `ET`.
    text_matrix: Matrix,
    line_matrix: Matrix,
    /// Fonts read, by the font dictionary's object number.
    fonts: HashMap<u32, Rc<Font>>,
    out: Content,
}

impl File {
    /// What `page` draws.
    ///
    /// # Errors
    ///
    /// Refuses page contents that are not streams; an operation that cannot be read is a warning instead.
    pub fn interpret(&self, page: &Page) -> Result<Content> {
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
                return Err(Error::new(format!(
                    "page {}: /Contents is a {}",
                    page.number,
                    other.kind()
                )));
            }
        }
        let mut interpreter = Interpreter {
            file: self,
            state: State::new(page.to_page_space()),
            stack: Vec::new(),
            path: Vec::new(),
            current: None,
            start: None,
            text_matrix: Matrix::IDENTITY,
            line_matrix: Matrix::IDENTITY,
            fonts: HashMap::new(),
            out: Content::default(),
        };
        interpreter.run(&data, &page.resources, 0);
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
        .ok_or_else(|| Error::new(format!("needs {count} numbers")))
}

/// The last operand, a string.
fn string(operands: &[Object]) -> Result<&[u8]> {
    match operands.last() {
        Some(Object::String(s)) => Ok(s),
        _ => Err(Error::new("needs a string")),
    }
}

/// The last operand, an integer (a style code is whole).
fn integer(operands: &[Object]) -> Result<i64> {
    operands
        .last()
        .and_then(Object::as_integer)
        .ok_or_else(|| Error::new("needs an integer"))
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
            _ => {} // text, clipping, shading, marked content and compatibility operators draw no path or image
        }
        Ok(())
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
                    .ok_or_else(|| Error::new("needs 6 numbers"))?;
                self.state.ctm = matrix.then(self.state.ctm);
            }
            b"w" => self.state.line_width = last(operands, 1)?[0],
            b"J" => self.state.cap = integer(operands)?,
            b"j" => self.state.join = integer(operands)?,
            b"d" => {
                let array = operands
                    .first()
                    .and_then(Object::as_array)
                    .ok_or_else(|| Error::new("needs a dash array"))?;
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
                let c1 = self.current.ok_or_else(|| Error::new("no current point"))?;
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
            b"BT" => (self.text_matrix, self.line_matrix) = (Matrix::IDENTITY, Matrix::IDENTITY),
            b"ET" => {}
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
                    .ok_or_else(|| Error::new("needs a font name"))?;
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
                    .ok_or_else(|| Error::new("needs 6 numbers"))?;
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
                    .ok_or_else(|| Error::new("needs an array"))?;
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
            return Err(Error::new("no /Font resources"));
        };
        let reference = fonts
            .get(name)
            .ok_or_else(|| Error::new(format!("no font /{}", String::from_utf8_lossy(name))))?;
        let number = reference.as_reference().map(|r| r.num);
        if let Some(font) = number.and_then(|n| self.fonts.get(&n)) {
            return Ok(Rc::clone(font));
        }
        let Object::Dict(dict) = self.file.resolve(reference)? else {
            return Err(Error::new(format!(
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
                bbox: Rect::around(&corners).expect("four corners"),
            });
            self.advance(advance);
        }
        out
    }

    fn text_event(&mut self, glyphs: Vec<Glyph>) {
        let Some(font) = &self.state.text.font else {
            return;
        };
        if glyphs.is_empty() {
            return;
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
            b"SC" | b"SCN" => (
                self.state.stroke_space.clone(),
                self.state.stroke_space.color(numbers_in(operands)),
            ),
            b"sc" | b"scn" => (
                self.state.fill_space.clone(),
                self.state.fill_space.color(numbers_in(operands)),
            ),
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
        self.path.clear();
        (self.current, self.start) = (None, None);
    }

    fn paint(&mut self, stroke: bool, fill: Option<bool>) {
        if !self.path.is_empty() {
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
            };
            self.out.events.push(Event::Path(event));
        }
        self.end_path();
    }

    fn ext_state(&mut self, operands: &[Object], resources: &Dict) -> Result<()> {
        let name = operands
            .first()
            .and_then(Object::as_name)
            .ok_or_else(|| Error::new("needs a name"))?;
        let Some(Object::Dict(states)) = self.file.get(resources, b"ExtGState")? else {
            return Err(Error::new("no /ExtGState resources"));
        };
        let Some(Object::Dict(gs)) = states.get(name).map(|o| self.file.resolve(o)).transpose()?
        else {
            return Err(Error::new(format!(
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
            .ok_or_else(|| Error::new("needs a name"))?;
        let Some(Object::Dict(objects)) = self.file.get(resources, b"XObject")? else {
            return Err(Error::new("no /XObject resources"));
        };
        let reference = objects
            .get(name)
            .ok_or_else(|| Error::new(format!("no XObject /{}", String::from_utf8_lossy(name))))?;
        let Object::Stream(stream) = self.file.resolve(reference)? else {
            return Err(Error::new(format!(
                "XObject /{} is not a stream",
                String::from_utf8_lossy(name)
            )));
        };
        let subtype = stream
            .dict
            .get(b"Subtype")
            .and_then(Object::as_name)
            .unwrap_or(b"");
        match subtype {
            b"Image" => {
                let int = |k: &[u8]| stream.dict.get(k).and_then(Object::as_integer).unwrap_or(0);
                self.image(
                    int(b"Width"),
                    int(b"Height"),
                    reference.as_reference().map(|r| r.num),
                    None,
                );
            }
            b"Form" => {
                if depth >= MAX_FORM_DEPTH {
                    return Err(Error::new("Form XObjects nest too deep"));
                }
                let matrix = match self.file.get(&stream.dict, b"Matrix")? {
                    Some(Object::Array(m)) => Matrix::from_numbers(
                        &m.iter().filter_map(Object::as_number).collect::<Vec<_>>(),
                    )
                    .unwrap_or(Matrix::IDENTITY),
                    _ => Matrix::IDENTITY,
                };
                let own = match self.file.get(&stream.dict, b"Resources")? {
                    Some(Object::Dict(r)) => r,
                    _ => resources.clone(),
                };
                let data = self.file.decode(&stream)?.into_bytes()?;
                let saved = (
                    self.state.clone(),
                    self.stack.len(),
                    std::mem::take(&mut self.path),
                    self.current,
                    self.start,
                );
                self.state.ctm = matrix.then(self.state.ctm);
                self.run(&data, &own, depth + 1);
                self.state = saved.0;
                self.stack.truncate(saved.1);
                (self.path, self.current, self.start) = (saved.2, saved.3, saved.4);
            }
            _ => {} // PostScript XObjects draw nothing a reader sees
        }
        Ok(())
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
        self.out.events.push(Event::Image(ImageEvent {
            bbox,
            matrix: m,
            width,
            height,
            object,
            inline,
        }));
    }
}

fn numbers_in(operands: &[Object]) -> Vec<f64> {
    operands.iter().filter_map(Object::as_number).collect()
}
