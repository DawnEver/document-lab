//! Glyphs to spans: runs of text in one font and size along one baseline, the unit a page's text is read in.
//!
//! Consecutive glyphs join while their font, size and writing direction hold and each starts where the last
//! ended: on its baseline (within [`BASELINE`] sizes) and no further on than [`BREAK`] sizes. A gap wider than
//! [`SPACE`] sizes with no space glyph in it reads as a space, as the eye reads it. Text events are taken in
//! drawing order, so a span never joins glyphs drawn apart.

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
}

struct Open {
    span: Span,
    end: Point,
    direction: (f64, f64),
    /// Whether a glyph with ink (not white-space) is in it yet: only those make its box.
    inked: bool,
}

/// The spans of a page's events, in drawing order.
#[must_use]
pub fn spans(events: &[Event]) -> Vec<Span> {
    let mut out: Vec<Span> = Vec::new();
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
                open.as_ref().map_or((1.0, 0.0), |o| o.direction)
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
            if let Some(done) = open.take() {
                out.push(done.span);
            }
            open = Some(Open {
                span: Span {
                    text: piece.to_string(),
                    font: Rc::clone(&text.font),
                    size: text.size,
                    bbox: glyph.bbox,
                    render_mode: text.render_mode,
                },
                end: glyph.end,
                direction,
                inked: !piece.trim().is_empty(),
            });
        }
    }
    out.extend(open.map(|o| o.span));
    out.retain(|s| !s.text.trim().is_empty());
    for span in &mut out {
        let trimmed = span.text.trim_end().len();
        span.text.truncate(trimmed);
    }
    out
}

/// Whether the glyph at `origin` continues `open`: None when it starts another span, else whether a space lies
/// before it.
fn join(open: &Open, text: &TextEvent, origin: Point, direction: (f64, f64)) -> Option<bool> {
    let size = open.span.size.max(f64::MIN_POSITIVE);
    if open.span.font != text.font
        || (open.span.size - text.size).abs() > 0.01 * size
        || open.span.render_mode != text.render_mode
        || (open.direction.0 - direction.0).abs() + (open.direction.1 - direction.1).abs() > 0.01
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

fn union(a: Rect, b: Rect) -> Rect {
    Rect {
        x0: a.x0.min(b.x0),
        y0: a.y0.min(b.y0),
        x1: a.x1.max(b.x1),
        y1: a.y1.max(b.y1),
    }
}
