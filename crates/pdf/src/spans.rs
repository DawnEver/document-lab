//! Glyphs to spans: runs of text in one font and size along one baseline -- or down one column, when the font writes
//! vertically -- the unit a page's text is read in.
//!
//! Consecutive glyphs join while their font, size, fill, writing mode and direction hold and each starts where the
//! last ended: on its baseline (within [`BASELINE`] sizes) and no further on than [`BREAK`] sizes. A gap wider than
//! [`SPACE`] sizes with no space glyph in it reads as a space, as the eye reads it. Text events are taken in
//! drawing order, so a span never joins glyphs drawn apart, nor glyphs of two layers (hidden, an annotation's), and
//! a vertical span's text reads top to bottom.
//!
//! Where a word gap is, is this module's alone. A span continues the word of another that ends at most [`SPACE`]
//! sizes before its start, raised or lowered by no more than [`RISE`] sizes, no space glyph between -- a change
//! of font, size or baseline inside a word -- whenever either was drawn (a subscript is often drawn after the
//! formula it sits in). Spans continuing one another share a number ([`Span::unspaced`]), so a reader setting
//! spans into lines spaces them by that and measures no gap of its own.
//!
//! An accent drawn as a glyph of its own over (or under) a letter, as TeX without T1 encoding draws `ü` (`u` and
//! `¨`), is one glyph with it before spans are made: the two composed as [`crate::accents`] says, the letter's
//! place in the drawing order kept, its box spanning both. One set on no letter stays as it is (a lone `¨`).

use std::borrow::Cow;
use std::rc::Rc;

use crate::accents;
use crate::color::Color;
use crate::font::FontInfo;
use crate::geometry::{Point, Rect};
use crate::interpret::{Event, Layer, TextEvent};

/// A gap wider than this many font sizes between two glyphs is a space.
pub const SPACE: f64 = 0.15;
/// A glyph further on than this many font sizes (or back by more than half) starts another span.
pub const BREAK: f64 = 1.0;
/// A glyph whose baseline is off the span's by more than this many font sizes starts another span.
pub const BASELINE: f64 = 0.1;
/// A glyph raised or lowered by up to this many font sizes (a superscript, a subscript) still continues the word
/// before it.
pub const RISE: f64 = 0.5;

/// A run of text.
#[derive(Debug, Clone, PartialEq)]
pub struct Span {
    /// Its text; a glyph no source names is U+FFFD.
    pub text: String,
    /// Its font.
    pub font: Rc<FontInfo>,
    /// The font size in page space.
    pub size: f64,
    /// The union of its glyphs' boxes, in page space.
    pub bbox: Rect,
    /// The text rendering mode it was shown in (3: invisible, as an OCR text layer is).
    pub render_mode: i64,
    /// The layer its glyphs are drawn in: a span never joins glyphs of two.
    pub layer: Layer,
    /// Whether it is written vertically (its font's writing mode): its glyphs stand upright and step down its
    /// column, its text reading top to bottom.
    pub vertical: bool,
    /// The unit vector its baseline runs along, in page space (y down): `(1, 0)` for text set left to right,
    /// `(0, -1)` for text set bottom to top.
    pub direction: (f64, f64),
    /// The number it shares, on its page, with the spans it continues or that continue it, no word gap between:
    /// a change of font, size or baseline inside a word (`Head` set bold in `Headline`; the `2` of `H2O`). None
    /// when it continues no span and none continues it.
    pub unspaced: Option<usize>,
    /// The color its glyphs are filled with; None when its rendering mode does not fill (stroke, invisible, clip).
    pub fill: Option<Color>,
}

/// The fill of glyphs shown in `text`: its fill color when its rendering mode fills (0, 2, 4, 6).
fn fill_of(text: &TextEvent) -> Option<Color> {
    matches!(text.render_mode, 0 | 2 | 4 | 6).then(|| text.fill.clone())
}

struct Open {
    span: Span,
    /// Where its first glyph starts.
    origin: Point,
    end: Point,
    /// Whether a glyph with ink (not white-space) is in it yet: only those make its box.
    inked: bool,
}

/// A glyph as spans read it: its text event, its text (U+FFFD when no source names it), where it starts and ends,
/// and its box.
struct Placed<'a> {
    event: &'a TextEvent,
    text: Cow<'a, str>,
    origin: Point,
    end: Point,
    bbox: Rect,
}

/// The spans of a page's events, in drawing order.
#[must_use]
pub fn spans(events: &[Event]) -> Vec<Span> {
    let mut glyphs: Vec<Placed> = events
        .iter()
        .filter_map(|event| match event {
            Event::Text(text) => Some(text),
            _ => None,
        })
        .flat_map(|event| {
            event.glyphs.iter().map(move |glyph| Placed {
                event,
                text: Cow::Borrowed(glyph.text.as_deref().unwrap_or("\u{FFFD}")),
                origin: glyph.origin,
                end: glyph.end,
                bbox: glyph.bbox,
            })
        })
        .collect();
    compose_accents(&mut glyphs);
    let mut out: Vec<Open> = Vec::new();
    let mut open: Option<Open> = None;
    for glyph in &glyphs {
        let text = glyph.event;
        let length = (glyph.end.x - glyph.origin.x).hypot(glyph.end.y - glyph.origin.y);
        let direction = if length > 0.0 {
            (
                (glyph.end.x - glyph.origin.x) / length,
                (glyph.end.y - glyph.origin.y) / length,
            )
        } else {
            let along = if text.vertical {
                (0.0, 1.0)
            } else {
                (1.0, 0.0)
            };
            open.as_ref().map_or(along, |o| o.span.direction)
        };
        let piece = glyph.text.as_ref();
        let joined = open
            .as_mut()
            .and_then(|o| join(o, text, glyph.origin, direction).map(|gap| (o, gap)));
        if let Some((o, gap)) = joined {
            if gap && !o.span.text.ends_with(' ') && !piece.starts_with(' ') {
                o.span.text.push(' ');
            }
            o.span.text.push_str(piece);
            let ink = inks(piece);
            if ink {
                o.span.bbox = if o.inked {
                    o.span.bbox.union(glyph.bbox)
                } else {
                    glyph.bbox
                };
                o.inked = true;
            }
            o.end = glyph.end;
            continue;
        }
        out.extend(open.take());
        open = Some(Open {
            span: Span {
                text: piece.to_string(),
                font: Rc::clone(&text.font.info),
                size: text.size,
                bbox: glyph.bbox,
                render_mode: text.render_mode,
                layer: text.layer.clone(),
                vertical: text.vertical,
                direction,
                unspaced: None,
                fill: fill_of(text),
            },
            origin: glyph.origin,
            end: glyph.end,
            inked: inks(piece),
        });
    }
    out.extend(open);
    let unspaced = unspaced(&out);
    out.into_iter()
        .zip(unspaced)
        .filter(|(o, _)| !o.span.text.trim().is_empty())
        .map(|(o, unspaced)| {
            let mut span = o.span;
            let trimmed = span.text.trim_end().len();
            span.text.truncate(trimmed);
            Span { unspaced, ..span }
        })
        .collect()
}

/// Each span's [`Span::unspaced`] number: the lowest index among the spans linked to it by continuing one
/// another's words; None when it is linked to none.
fn unspaced(spans: &[Open]) -> Vec<Option<usize>> {
    fn root(parent: &mut [usize], mut i: usize) -> usize {
        while parent[i] != i {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        i
    }
    let mut parent: Vec<usize> = (0..spans.len()).collect();
    let mut linked = vec![false; spans.len()];
    for (i, next) in spans.iter().enumerate() {
        for (j, before) in spans.iter().enumerate() {
            if i != j && continues_word(before, next) {
                let (a, b) = (root(&mut parent, i), root(&mut parent, j));
                parent[a.max(b)] = a.min(b);
                linked[i] = true;
                linked[j] = true;
            }
        }
    }
    (0..spans.len())
        .map(|i| linked[i].then(|| root(&mut parent, i)))
        .collect()
}

/// Each accent glyph set on a letter merged into it ([`sits_on`]): the letter's text becomes the two composed
/// ([`accents::compose`]), its box the union of theirs; the accent is gone. An accent tries the nearest letter
/// drawn after it (TeX draws the accent first), then the nearest drawn before it, other accents passed over; the
/// last accents are taken first, so of two stacked on one letter the inner composes first (`\'{\"u}` is `ǘ`).
fn compose_accents(glyphs: &mut Vec<Placed>) {
    let marks: Vec<Option<char>> = glyphs.iter().map(|g| accents::mark(&g.text)).collect();
    if marks.iter().all(Option::is_none) {
        return;
    }
    let mut gone = vec![false; glyphs.len()];
    for i in (0..glyphs.len()).rev() {
        let Some(mark) = marks[i] else { continue };
        let letter = |j: &usize| !gone[*j] && marks[*j].is_none();
        let after = (i + 1..glyphs.len()).find(letter);
        let before = (0..i).rev().find(letter);
        for j in [after, before].into_iter().flatten() {
            if !sits_on(&glyphs[i], &glyphs[j]) {
                continue;
            }
            if let Some(text) = accents::compose(&glyphs[j].text, mark) {
                glyphs[j].text = Cow::Owned(text);
                glyphs[j].bbox = glyphs[j].bbox.union(glyphs[i].bbox);
                gone[i] = true;
                break;
            }
        }
    }
    let mut gone = gone.into_iter();
    glyphs.retain(|_| !gone.next().unwrap_or(false));
}

/// Whether `accent` is set on `letter`: in its layer, the middle of its advance within the letter's, its baseline
/// the letter's -- raised by no more than [`RISE`] sizes, as TeX raises an accent over a capital, or lowered by no
/// more than [`BASELINE`] -- along the letter's direction. (An accent glyph is drawn above or below the baseline it
/// stands on: the glyph boxes, from the font's descent to its ascent, do not tell where its ink is.)
fn sits_on(accent: &Placed, letter: &Placed) -> bool {
    let (dx, dy) = (
        letter.end.x - letter.origin.x,
        letter.end.y - letter.origin.y,
    );
    let advance = dx.hypot(dy);
    if advance <= 0.0 || accent.event.layer != letter.event.layer {
        return false;
    }
    let (ux, uy) = (dx / advance, dy / advance);
    let middle = Point {
        x: f64::midpoint(accent.origin.x, accent.end.x) - letter.origin.x,
        y: f64::midpoint(accent.origin.y, accent.end.y) - letter.origin.y,
    };
    let (ox, oy) = (
        accent.origin.x - letter.origin.x,
        accent.origin.y - letter.origin.y,
    );
    let size = letter.event.size;
    let rise = ox * uy - oy * ux; // up from the letter's baseline (page space grows down)
    (0.0..=advance).contains(&(middle.x * ux + middle.y * uy))
        && (-BASELINE * size..=RISE * size).contains(&rise)
}

/// Whether the glyph at `origin` continues `open`: None when it starts another span, else whether a space lies
/// before it.
fn join(open: &Open, text: &TextEvent, origin: Point, direction: (f64, f64)) -> Option<bool> {
    let size = open.span.size.max(f64::MIN_POSITIVE);
    if open.span.font != text.font.info
        || (open.span.size - text.size).abs() > 0.01 * size
        || open.span.render_mode != text.render_mode
        || open.span.fill != fill_of(text)
        || open.span.layer != text.layer
        || open.span.vertical != text.vertical
        || (open.span.direction.0 - direction.0).abs() + (open.span.direction.1 - direction.1).abs()
            > 0.01
    {
        return None;
    }
    let (dx, dy) = (origin.x - open.end.x, origin.y - open.end.y);
    let along = dx * direction.0 + dy * direction.1;
    let across = -dx * direction.1 + dy * direction.0;
    if across.abs() > BASELINE * size || along > BREAK * size || along < -0.5 * size {
        return None;
    }
    Some(along > SPACE * size)
}

/// Whether a glyph whose text is `piece` makes its span's box: it shows ink -- its text is not white-space, or is
/// empty because an `/ActualText` before it in its sequence carries its text.
fn inks(piece: &str) -> bool {
    piece.is_empty() || !piece.trim().is_empty()
}

/// Whether `next` continues the word `before` ends in: no space glyph between them, both written the same way (one
/// writing mode, one direction) in one layer, `next` starting on from `before`'s end by no more than a word gap
/// (back by no more than half a size), raised or lowered by no more than [`RISE`] sizes.
fn continues_word(before: &Open, next: &Open) -> bool {
    if before.span.vertical != next.span.vertical
        || before.span.text.ends_with(char::is_whitespace)
        || next.span.text.starts_with(char::is_whitespace)
        || before.span.layer != next.span.layer
    {
        return false;
    }
    let size = before.span.size.max(next.span.size).max(f64::MIN_POSITIVE);
    let (dx, dy) = before.span.direction;
    let along = |from: Point| (next.origin.x - from.x) * dx + (next.origin.y - from.y) * dy;
    let across = -(next.origin.x - before.end.x) * dy + (next.origin.y - before.end.y) * dx;
    (dx - next.span.direction.0).abs() + (dy - next.span.direction.1).abs() <= 0.01
        && (-0.5 * size..=SPACE * size).contains(&along(before.end))
        && across.abs() <= RISE * size
}
