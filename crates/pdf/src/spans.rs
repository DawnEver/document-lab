//! Glyphs to spans: runs of text in one font and size along one baseline -- or down one column, when the font writes
//! vertically -- the unit a page's text is read in.
//!
//! Consecutive glyphs join while their font, size, writing mode and direction hold and each starts where the last
//! ended: on its baseline (within [`BASELINE`] sizes) and no further on than [`BREAK`] sizes. A gap wider than
//! [`SPACE`] sizes with no space glyph in it reads as a space, as the eye reads it. Text events are taken in
//! drawing order, so a span never joins glyphs drawn apart, and a vertical span's text reads top to bottom.
//!
//! Where a word gap is, is this module's alone. A span continues the word of another that ends at most [`SPACE`]
//! sizes before its start, raised or lowered by no more than [`RISE`] sizes, no space glyph between -- a change
//! of font, size or baseline inside a word -- whenever either was drawn (a subscript is often drawn after the
//! formula it sits in). Spans continuing one another share a number ([`Span::unspaced`]), so a reader setting
//! spans into lines spaces them by that and measures no gap of its own.

use std::rc::Rc;

use crate::font::FontInfo;
use crate::geometry::{Point, Rect};
use crate::interpret::{Event, TextEvent};

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
}

struct Open {
    span: Span,
    /// Where its first glyph starts.
    origin: Point,
    end: Point,
    /// Whether a glyph with ink (not white-space) is in it yet: only those make its box.
    inked: bool,
}

/// The spans of a page's events, in drawing order.
#[must_use]
pub fn spans(events: &[Event]) -> Vec<Span> {
    let mut out: Vec<Open> = Vec::new();
    let mut open: Option<Open> = None;
    for event in events {
        let Event::Text(text) = event else { continue };
        for glyph in &text.glyphs {
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
            let piece = glyph.text.as_deref().unwrap_or("\u{FFFD}");
            let joined = open
                .as_mut()
                .and_then(|o| join(o, text, glyph.origin, direction).map(|gap| (o, gap)));
            if let Some((o, gap)) = joined {
                if gap && !o.span.text.ends_with(' ') && !piece.starts_with(' ') {
                    o.span.text.push(' ');
                }
                o.span.text.push_str(piece);
                let ink = !piece.trim().is_empty();
                if ink {
                    o.span.bbox = if o.inked {
                        union(o.span.bbox, glyph.bbox)
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
                    font: Rc::clone(&text.font),
                    size: text.size,
                    bbox: glyph.bbox,
                    render_mode: text.render_mode,
                    vertical: text.vertical,
                    direction,
                    unspaced: None,
                },
                origin: glyph.origin,
                end: glyph.end,
                inked: !piece.trim().is_empty(),
            });
        }
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

/// Whether the glyph at `origin` continues `open`: None when it starts another span, else whether a space lies
/// before it.
fn join(open: &Open, text: &TextEvent, origin: Point, direction: (f64, f64)) -> Option<bool> {
    let size = open.span.size.max(f64::MIN_POSITIVE);
    if open.span.font != text.font
        || (open.span.size - text.size).abs() > 0.01 * size
        || open.span.render_mode != text.render_mode
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

/// Whether `next` continues the word `before` ends in: no space glyph between them, both written the same way (one
/// writing mode, one direction), `next` starting on from `before`'s end by no more than a word gap (back by no more
/// than half a size), raised or lowered by no more than [`RISE`] sizes.
fn continues_word(before: &Open, next: &Open) -> bool {
    if before.span.vertical != next.span.vertical
        || before.span.text.ends_with(char::is_whitespace)
        || next.span.text.starts_with(char::is_whitespace)
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

fn union(a: Rect, b: Rect) -> Rect {
    Rect {
        x0: a.x0.min(b.x0),
        y0: a.y0.min(b.y0),
        x1: a.x1.max(b.x1),
        y1: a.y1.max(b.y1),
    }
}
