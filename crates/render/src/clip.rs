//! Clipping paths to masks: what of the canvas the chain of clips in force leaves (ISO 32000-2 section 8.5.4), each
//! path filled anti-aliased by its rule, a chain the product of its paths' coverage.
//!
//! A rectangle square to the canvas clips to the whole pixels it touches, as a grid-fitted image covers them
//! (`crate::image`; measured against `MuPDF`, which does the same); a clip holding the whole canvas is no clip, so
//! drawing under it needs no mask. A chain is made once: the events drawn under one clip share it, and a chain whose
//! paths are the same as one made before (a figure clipping each of its curves to the same plot box) is that one's.
//! A text clip is its glyphs' outlines, filled by the nonzero rule; a glyph with no outline to read (a Type 3 glyph, a
//! font with no program) stands in by its box, counted as such.

use std::collections::hash_map::DefaultHasher;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::rc::Rc;

use pdf::geometry::Segment;
use pdf::geometry::{Point, Rect};
use pdf::interpret::{Clip, Outline};
use tiny_skia::{FillRule, Mask, PathBuilder, Transform};

use crate::Grid;

/// How many distinct chains' masks are kept.
const KEPT: usize = 16;

/// What of the canvas a chain of clips leaves.
#[derive(Clone)]
pub enum Left {
    /// All of it.
    All,
    /// None of it.
    Nothing,
    /// Each pixel's coverage.
    Mask(Rc<Mask>),
}

/// The masks of the clips met so far.
#[derive(Default)]
pub struct Clips {
    /// Each chain met, by its innermost clip (kept alive with it, so its address names it alone): its content's key.
    keys: HashMap<*const Clip, (Rc<Clip>, u64)>,
    /// What each chain's content leaves, the most recent [`KEPT`].
    made: HashMap<u64, Left>,
    order: VecDeque<u64>,
}

impl Clips {
    /// What `clip` leaves of the canvas `grid` maps onto. A text clip holding a glyph by its box is counted in
    /// `undrawn`.
    pub(crate) fn mask(
        &mut self,
        clip: Option<&Rc<Clip>>,
        grid: &Grid,
        undrawn: &mut BTreeMap<String, usize>,
    ) -> Left {
        let Some(clip) = clip else {
            return Left::All;
        };
        if by_boxes(clip) {
            *undrawn
                .entry("text clip glyph with no outline (by its box)".to_string())
                .or_default() += 1;
        }
        self.left(clip, grid)
    }

    fn left(&mut self, clip: &Rc<Clip>, grid: &Grid) -> Left {
        let key = self.key(clip);
        if let Some(left) = self.made.get(&key) {
            return left.clone();
        }
        let within = match &clip.within {
            Some(outer) => self.left(outer, grid),
            None => Left::All,
        };
        let left = match within {
            Left::Nothing => Left::Nothing,
            within => narrowed(within, &clip.outline, grid),
        };
        if self.order.len() >= KEPT
            && let Some(oldest) = self.order.pop_front()
        {
            self.made.remove(&oldest);
        }
        self.order.push_back(key);
        self.made.insert(key, left.clone());
        left
    }

    /// The key of the chain `clip` ends: its outlines' content, hashed.
    fn key(&mut self, clip: &Rc<Clip>) -> u64 {
        let at = Rc::as_ptr(clip);
        if let Some((_, key)) = self.keys.get(&at) {
            return *key;
        }
        let outer = clip.within.as_ref().map_or(0, |o| self.key(o));
        let mut hasher = DefaultHasher::new();
        outer.hash(&mut hasher);
        std::mem::discriminant(&clip.outline).hash(&mut hasher);
        match &clip.outline {
            Outline::Path { segments, even_odd } => {
                even_odd.hash(&mut hasher);
                hash_segments(segments, &mut hasher);
            }
            Outline::Glyphs { segments, unshaped } => {
                hash_segments(segments, &mut hasher);
                for b in unshaped {
                    [b.x0, b.y0, b.x1, b.y1].map(f64::to_bits).hash(&mut hasher);
                }
            }
        }
        let key = hasher.finish();
        self.keys.insert(at, (Rc::clone(clip), key));
        key
    }
}

/// `segments`' content, hashed.
fn hash_segments(segments: &[Segment], hasher: &mut DefaultHasher) {
    for s in segments {
        let points: &[Point] = match s {
            Segment::MoveTo(p) | Segment::LineTo(p) => std::slice::from_ref(p),
            Segment::CurveTo(a, b, p) => &[*a, *b, *p],
            Segment::Close => &[],
        };
        std::mem::discriminant(s).hash(hasher);
        for p in points {
            (p.x.to_bits(), p.y.to_bits()).hash(hasher);
        }
    }
}

/// Whether a clip of the chain `clip` ends is a text clip holding a glyph by its box.
fn by_boxes(clip: &Clip) -> bool {
    matches!(&clip.outline, Outline::Glyphs { unshaped, .. } if !unshaped.is_empty())
        || clip.within.as_deref().is_some_and(by_boxes)
}

/// What `within` leaves, narrowed to `outline`.
fn narrowed(within: Left, outline: &Outline, grid: &Grid) -> Left {
    let (path, rule) = match outline {
        Outline::Path { segments, even_odd } => {
            if let Some(r) = rectangle(segments, grid) {
                let Some(snapped) = snapped(r, grid.bounds()) else {
                    return Left::Nothing;
                };
                if snapped == grid.bounds() {
                    return within;
                }
                (rect_path(&[snapped]), FillRule::Winding)
            } else {
                let rule = if *even_odd {
                    FillRule::EvenOdd
                } else {
                    FillRule::Winding
                };
                (crate::build(segments, &|p| grid.point(p)), rule)
            }
        }
        Outline::Glyphs { segments, unshaped } => {
            let mut builder = PathBuilder::new();
            if let Some(outlines) = crate::build(segments, &|p| grid.point(p)) {
                builder.push_path(&outlines);
            }
            let boxes: Vec<Rect> = unshaped.iter().map(|b| grid.rect(*b)).collect();
            if let Some(boxes) = rect_path(&boxes) {
                builder.push_path(&boxes);
            }
            (builder.finish(), FillRule::Winding)
        }
    };
    let Some(path) = path else {
        return Left::Nothing; // a clip of no area leaves nothing
    };
    let mask = if let Left::Mask(m) = within {
        let mut mask = (*m).clone();
        mask.intersect_path(&path, rule, true, Transform::identity());
        mask
    } else {
        let mut mask = Mask::new(grid.width, grid.height).expect("the canvas's size");
        mask.fill_path(&path, rule, true, Transform::identity());
        mask
    };
    if mask.data().iter().all(|&a| a == 0) {
        Left::Nothing
    } else {
        Left::Mask(Rc::new(mask))
    }
}

/// The device pixels (the page's) what is drawn under the chain `clip` ends may paint, as `MuPDF` bounds them -- its
/// scissor, where its image painter starts stepping ([`crate::image::sampled`]): the whole pixels each of the chain's
/// outlines' points lie within, and the canvas's scissor ([`Grid::scissor`]).
pub(crate) fn scissor(clip: Option<&Rc<Clip>>, grid: &Grid) -> Rect {
    let mut within = grid.scissor;
    let mut next = clip;
    while let Some(clip) = next {
        let points: Vec<Point> = match &clip.outline {
            Outline::Path { segments, .. } => crate::points(segments),
            Outline::Glyphs { segments, unshaped } => {
                let mut points = crate::points(segments);
                points.extend(
                    unshaped
                        .iter()
                        .flat_map(|b| [Point { x: b.x0, y: b.y0 }, Point { x: b.x1, y: b.y1 }]),
                );
                points
            }
        };
        let device: Vec<Point> = points
            .into_iter()
            .map(|p| {
                let q = grid.point(p);
                Point {
                    x: q.x + grid.left,
                    y: q.y + grid.top,
                }
            })
            .collect();
        if let Some(r) = Rect::around(&device) {
            within = within.intersect(Rect {
                x0: r.x0.floor(),
                y0: r.y0.floor(),
                x1: r.x1.ceil(),
                y1: r.y1.ceil(),
            });
        }
        next = clip.within.as_ref();
    }
    within
}

/// The whole pixels the path `segments` (page space) covers when it is one rectangle square to the canvas -- widened
/// to the whole pixels it touches, as such a clip is; None when it is not one, or covers no pixel.
pub(crate) fn whole_pixels(segments: &[Segment], grid: &Grid) -> Option<Rect> {
    snapped(rectangle(segments, grid)?, grid.bounds())
}

/// `r` (pixel space) widened to the whole pixels it touches, within `canvas`; None when it covers none.
fn snapped(r: Rect, canvas: Rect) -> Option<Rect> {
    const SNAP: f64 = 1e-3;
    let s = Rect {
        x0: (r.x0 + SNAP).floor().max(canvas.x0),
        y0: (r.y0 + SNAP).floor().max(canvas.y0),
        x1: (r.x1 - SNAP).ceil().min(canvas.x1),
        y1: (r.y1 - SNAP).ceil().min(canvas.y1),
    };
    (s.x0 < s.x1 && s.y0 < s.y1).then_some(s)
}

/// A path of the rectangles `boxes` (pixel space); None when none has an area.
fn rect_path(boxes: &[Rect]) -> Option<tiny_skia::Path> {
    let mut builder = PathBuilder::new();
    for b in boxes {
        #[expect(clippy::cast_possible_truncation, reason = "pixel coordinates")]
        if let Some(rect) =
            tiny_skia::Rect::from_ltrb(b.x0 as f32, b.y0 as f32, b.x1 as f32, b.y1 as f32)
        {
            builder.push_rect(rect);
        }
    }
    builder.finish()
}

/// The box `segments` are, in pixel space, when they are one rectangle square to the canvas.
fn rectangle(segments: &[Segment], grid: &Grid) -> Option<Rect> {
    let mut points = Vec::with_capacity(5);
    for (i, s) in segments.iter().enumerate() {
        match s {
            Segment::MoveTo(p) if i == 0 => points.push(grid.point(*p)),
            Segment::LineTo(p) => points.push(grid.point(*p)),
            Segment::Close => {}
            _ => return None,
        }
    }
    if !(4..=5).contains(&points.len()) {
        return None;
    }
    let r = Rect::around(&points)?;
    let near = |a: f64, b: f64| (a - b).abs() < 1e-6;
    let corners = points
        .iter()
        .all(|p| (near(p.x, r.x0) || near(p.x, r.x1)) && (near(p.y, r.y0) || near(p.y, r.y1)));
    // Consecutive corners differ in one coordinate alone: no bow-tie.
    let sides = points
        .windows(2)
        .all(|w| near(w[0].x, w[1].x) || near(w[0].y, w[1].y));
    (corners && sides).then_some(r)
}
