//! What a stroke paints, bounded from outside: never smaller than what it paints, so a reader that does not draw it
//! can still ask whether it may paint inside a box ([`Stroke::may_paint`]) and how far past its path it may reach
//! ([`Stroke::reach`]). Drawing it is the renderer's (`crates/render`, through `tiny-skia`'s stroker); this bound is
//! what both read when they only need to know where a stroke may reach.
//!
//! A stroke paints its band -- a straight piece swept across by half its line width either way (8.4.3.2) -- and
//! round its vertices its caps and joins (8.4.3.3, 8.4.3.4): a butt cap nothing, a round cap or join a disc of the
//! half width, a bevel join a triangle within that disc, a square cap a half square whose far corners lie `√2` half
//! widths from the end, a miter join a tip `1 / sin(φ/2)` half widths from its vertex, `φ` the angle between its
//! pieces, when that ratio is within the miter limit, else beveled. A dash paints part of the band and caps its own
//! ends, on the path: a butt or round cap inside the band but within a half width past a vertex, a square cap inside
//! it too but within `√2` half widths of a vertex it runs past -- on a curve, of the curve. A join or cap value the
//! PDF does not define is drawn as the renderer draws it: a miter, a butt.
//!
//! A line is stroked in user space (8.5.3.2): a straight piece's band is a rectangle there, a parallelogram on the
//! page, its half width across it in page space at most the user space one times the most the matrix stretches a
//! length ([`Matrix::stretch`](crate::geometry::Matrix::stretch)), and caps, joins and dashes scale with it. A
//! matrix that flattens the page (no area) leaves the user space width unknown: its reach is unbounded.
//!
//! So the bound, with `h` that greatest half width: every straight piece its band exactly; an open subpath's ends
//! their cap's reach; every join between two pieces its own (a miter's angle read in user space, from the pieces'
//! directions there -- a curve's its tangent --, the limit when a direction is unknown), or dashed its cap's when
//! farther; every curve within `h` (or, dashed, its cap's reach) of the curve, bounded by its chord -- a Bezier curve
//! lies in the convex hull of its control points, so within the farther inner control point's distance of its chord
//! -- and split at its middle (de Casteljau) until that bound settles whether it reaches the box, or [`SPLITS`] deep
//! says it may.

use crate::geometry::{Point, Rect, Segment, crosses, touches};
use crate::interpret::Stroke;

/// How many times a curve is halved before a bound that still does not settle whether it reaches a box says it
/// may: its chord's bulge shrinks four times a split, so this is far past any page's precision.
pub const SPLITS: u32 = 24;

impl Stroke {
    /// The most its band reaches across its path, in page space: half its user space width times the most its
    /// matrix stretches a length; unbounded when its matrix flattens the page.
    #[must_use]
    pub fn half_width(&self) -> f64 {
        let scale = self.ctm.scale();
        let half = self.width / scale / 2.0 * self.ctm.stretch();
        if scale > 0.0 && half >= 0.0 {
            half
        } else {
            f64::INFINITY
        }
    }

    /// The most any of its paint reaches from its path, in page space, whatever the path: its half width, a square
    /// cap's corner or a miter join's tip at its limit the furthest.
    #[must_use]
    pub fn reach(&self) -> f64 {
        let reach = Reach::of(self);
        let miter = if matches!(self.join, 1 | 2) {
            1.0
        } else {
            self.miter_limit.max(1.0)
        };
        reach.cap.max(reach.half * miter)
    }

    /// Whether stroking `segments` (page space) may paint inside `rect` -- share some area with it; along its edge
    /// does not --: never false where it paints, within rounding (see the module).
    #[must_use]
    pub fn may_paint(&self, segments: &[Segment], rect: Rect) -> bool {
        let reach = Reach::of(self);
        if reach.half.is_infinite() {
            return true;
        }
        let mut subpath = Subpath::default();
        for segment in segments {
            if subpath.closed && !matches!(segment, Segment::MoveTo(_)) {
                // A segment after a close starts a subpath at the closed one's start (8.5.2.1).
                if reach.round(&subpath, rect) {
                    return true;
                }
                subpath = Subpath::at(subpath.start);
            }
            match *segment {
                Segment::MoveTo(p) => {
                    if reach.round(&subpath, rect) {
                        return true;
                    }
                    subpath = Subpath::at(Some(p));
                }
                Segment::LineTo(p) => {
                    let a = subpath.at.unwrap_or(p);
                    if self.band(a, p, rect) {
                        return true;
                    }
                    subpath.add(a, p, p, a);
                }
                Segment::CurveTo(b, c, d) => {
                    let a = subpath.at.unwrap_or(b);
                    if curved([a, b, c, d], rect, reach.curve, SPLITS) {
                        return true;
                    }
                    // Its tangents: toward the first control point apart from its start, from the last apart from its
                    // end.
                    let leaves = [b, c, d].into_iter().find(|&q| q != a).unwrap_or(d);
                    let arrives = [c, b, a].into_iter().find(|&q| q != d).unwrap_or(a);
                    subpath.add(a, d, leaves, arrives);
                }
                Segment::Close => {
                    if let Some((a, start)) = subpath.at.zip(subpath.start) {
                        if self.band(a, start, rect) {
                            return true;
                        }
                        subpath.add(a, start, start, a);
                        subpath.closed = true;
                    }
                }
            }
        }
        reach.round(&subpath, rect)
    }

    /// Whether the straight piece from `a` to `b`, stroked, shares some area with `rect` (touching does not): its
    /// band, in user space the piece swept across by half the line width either way, a parallelogram on the page.
    /// Apart when one of the four axes the two shapes' edges are normal to parts them. A piece of no length has no
    /// band.
    fn band(&self, a: Point, b: Point, rect: Rect) -> bool {
        let [xx, xy, yx, yy, ..] = self.ctm.0;
        let along = Point {
            x: b.x - a.x,
            y: b.y - a.y,
        };
        let Some(user) = self.user(along) else {
            return false;
        };
        let user_half = self.width / self.ctm.scale() / 2.0;
        let (nx, ny) = (-user.y * user_half, user.x * user_half);
        let across = Point {
            x: xx * nx + yx * ny,
            y: xy * nx + yy * ny,
        };
        let axes = [
            Point { x: 1.0, y: 0.0 },
            Point { x: 0.0, y: 1.0 },
            Point {
                x: -along.y,
                y: along.x,
            },
            Point {
                x: -across.y,
                y: across.x,
            },
        ];
        let corners = rect.corners();
        !axes.iter().filter(|k| k.x != 0.0 || k.y != 0.0).any(|&k| {
            let dot = |p: Point| p.x * k.x + p.y * k.y;
            let (base, run, side) = (dot(a), dot(along), dot(across).abs());
            let (lo, hi) = (base + run.min(0.0) - side, base + run.max(0.0) + side);
            let (rlo, rhi) = corners
                .iter()
                .map(|&c| dot(c))
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), d| {
                    (l.min(d), h.max(d))
                });
            hi <= rlo || rhi <= lo
        })
    }

    /// The unit direction in user space of the page space direction `along` -- up to its sign, the same for every
    /// direction under one matrix --; None for no direction.
    fn user(&self, along: Point) -> Option<Point> {
        let [xx, xy, yx, yy, ..] = self.ctm.0;
        // The adjugate of the matrix's linear part: its inverse times its determinant.
        let (x, y) = (yy * along.x - yx * along.y, xx * along.y - xy * along.x);
        let length = x.hypot(y);
        (length > 0.0 && length.is_finite()).then(|| Point {
            x: x / length,
            y: y / length,
        })
    }

    /// How far its join reaches from a vertex the path arrives at going `arrives` and leaves going `leaves` (page
    /// space): a miter's tip, half the line width over the sine of half the angle between the pieces when that ratio
    /// is within the limit, else beveled (8.4.3.4); the limit itself when a direction is unknown; a round or bevel
    /// join half the width.
    fn join(&self, half: f64, arrives: Point, leaves: Point) -> f64 {
        if matches!(self.join, 1 | 2) {
            return half;
        }
        let limit = self.miter_limit.max(1.0);
        let Some((a, b)) = self.user(arrives).zip(self.user(leaves)) else {
            return half * limit;
        };
        // sin(phi / 2), phi the angle between the pieces, is cos(theta / 2), theta the turn from one to the next.
        let sine = ((1.0 + a.x * b.x + a.y * b.y) / 2.0).max(0.0).sqrt();
        // The tip's ratio within the limit, rounding counted the miter's way.
        if sine * limit * (1.0 + 1e-9) >= 1.0 {
            half / sine.min(1.0)
        } else {
            half
        }
    }
}

/// A subpath as its caps and joins see it: its start and current point, the pieces of some length it has drawn --
/// each its two ends and the directions leaving the first and reaching the second, page space -- and whether it is
/// closed.
#[derive(Default)]
struct Subpath {
    start: Option<Point>,
    at: Option<Point>,
    pieces: Vec<[Point; 4]>,
    closed: bool,
}

impl Subpath {
    /// A subpath started at `start`.
    fn at(start: Option<Point>) -> Self {
        Self {
            start,
            at: start,
            ..Self::default()
        }
    }

    /// The piece from `a` to `b`, leaving `a` toward `leaves` and reaching `b` from `arrives`: kept when it has a
    /// length.
    fn add(&mut self, a: Point, b: Point, leaves: Point, arrives: Point) {
        self.start = self.start.or(Some(a));
        self.at = Some(b);
        let gone = |p: Point, q: Point| Point {
            x: q.x - p.x,
            y: q.y - p.y,
        };
        if a != b || leaves != a {
            self.pieces.push([a, b, gone(a, leaves), gone(arrives, b)]);
        }
    }
}

/// How far, in page space, a stroke paints from its path: across a straight piece (`half`), from a curve (`curve`),
/// round an open subpath's end (`cap`) and, dashed, round any vertex a dash may end at (`dash`, else 0).
struct Reach<'a> {
    stroke: &'a Stroke,
    half: f64,
    curve: f64,
    cap: f64,
    dash: f64,
}

impl<'a> Reach<'a> {
    /// The reach of `stroke`: a dash may end anywhere, so dashed its cap reaches from a curve and every vertex too.
    fn of(stroke: &'a Stroke) -> Self {
        let half = stroke.half_width();
        let dashed = !stroke.dashes.is_empty();
        let cap = half
            * match stroke.cap {
                1 => 1.0,
                2 => std::f64::consts::SQRT_2,
                _ => 0.0,
            };
        Self {
            stroke,
            half,
            curve: if dashed { half.max(cap) } else { half },
            cap,
            dash: if dashed { cap } else { 0.0 },
        }
    }

    /// Whether `subpath`'s caps and joins reach inside `rect`: a join between each piece and the next (and, closed,
    /// the last and the first), an open one's ends its caps, a subpath of no length a dot -- its cap, round the
    /// one point it is at.
    fn round(&self, subpath: &Subpath, rect: Rect) -> bool {
        let near = |p: Point, reach: f64| distance(p, rect) < reach;
        let pieces = &subpath.pieces;
        let (Some(first), Some(last)) = (pieces.first(), pieces.last()) else {
            return subpath.start.is_some_and(|p| near(p, self.cap));
        };
        let joins = pieces.windows(2).map(|w| (w[0], w[1]));
        let closing = subpath.closed.then_some((*last, *first));
        let joined = joins
            .chain(closing)
            .any(|([_, at, _, arrives], [_, _, leaves, _])| {
                near(
                    at,
                    self.stroke.join(self.half, arrives, leaves).max(self.dash),
                )
            });
        joined || (!subpath.closed && (near(first[0], self.cap) || near(last[1], self.cap)))
    }
}

/// Whether the Bezier curve with `control` points, thickened by `half`, may share some area with `rect`: its
/// chord thickened by `half` and its bulge (the farther inner control point's distance from it) settles it, else
/// its middle point does, else its two halves do, `splits` more times at most.
fn curved(control: [Point; 4], rect: Rect, half: f64, splits: u32) -> bool {
    let [a, b, c, d] = control;
    let bulge = point_distance(b, a, d).max(point_distance(c, a, d));
    if !(crosses(a, d, rect) || segment_distance(a, d, rect) < half + bulge) {
        return false;
    }
    if splits == 0 {
        return true;
    }
    let mid = |from: Point, to: Point| Point {
        x: f64::midpoint(from.x, to.x),
        y: f64::midpoint(from.y, to.y),
    };
    let (ab, bc, cd) = (mid(a, b), mid(b, c), mid(c, d));
    let (abc, bcd) = (mid(ab, bc), mid(bc, cd));
    let middle = mid(abc, bcd);
    inside(middle, rect)
        || distance(middle, rect) < half
        || curved([a, ab, abc, middle], rect, half, splits - 1)
        || curved([middle, bcd, cd, d], rect, half, splits - 1)
}

/// Whether `p` lies inside `rect` (not on its edge).
fn inside(p: Point, rect: Rect) -> bool {
    rect.x0 < p.x && p.x < rect.x1 && rect.y0 < p.y && p.y < rect.y1
}

/// The distance from `p` to `rect` (0 on or inside it).
fn distance(p: Point, rect: Rect) -> f64 {
    let dx = (rect.x0 - p.x).max(p.x - rect.x1).max(0.0);
    let dy = (rect.y0 - p.y).max(p.y - rect.y1).max(0.0);
    dx.hypot(dy)
}

/// The distance from `p` to the segment from `a` to `b`.
fn point_distance(p: Point, a: Point, b: Point) -> f64 {
    let (dx, dy) = (b.x - a.x, b.y - a.y);
    let length = dx * dx + dy * dy;
    let t = if length > 0.0 {
        (((p.x - a.x) * dx + (p.y - a.y) * dy) / length).clamp(0.0, 1.0)
    } else {
        0.0
    };
    (a.x + t * dx - p.x).hypot(a.y + t * dy - p.y)
}

/// The distance from the segment from `a` to `b` to `rect` (0 where they meet): apart, the nearest points of a
/// segment and a convex polygon include an end of one of them.
fn segment_distance(a: Point, b: Point, rect: Rect) -> f64 {
    if touches(a, b, rect) {
        return 0.0;
    }
    rect.corners()
        .into_iter()
        .map(|c| point_distance(c, a, b))
        .chain([distance(a, rect), distance(b, rect)])
        .fold(f64::INFINITY, f64::min)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::Color;
    use crate::geometry::Matrix;

    /// A stroke `width` wide, its cap and join, its miter limit and its dashes, under `ctm`.
    fn stroke(
        width: f64,
        cap: i64,
        join: i64,
        miter_limit: f64,
        dashes: &[f64],
        ctm: Matrix,
    ) -> Stroke {
        Stroke {
            color: Color::black(),
            width,
            dashes: dashes.to_vec(),
            dash_phase: 0.0,
            cap,
            join,
            miter_limit,
            ctm,
        }
    }

    fn solid(width: f64, cap: i64, join: i64, miter_limit: f64) -> Stroke {
        stroke(width, cap, join, miter_limit, &[], Matrix::IDENTITY)
    }

    fn pt(x: f64, y: f64) -> Point {
        Point { x, y }
    }

    /// A line from `a` to `b`.
    fn line(a: Point, b: Point) -> Vec<Segment> {
        vec![Segment::MoveTo(a), Segment::LineTo(b)]
    }

    /// The box the tests' text is set in.
    const TEXT: Rect = Rect {
        x0: 100.0,
        y0: 100.0,
        x1: 200.0,
        y1: 110.0,
    };

    /// A frame drawn as `re` draws it, `margin` round [`TEXT`].
    fn frame(margin: f64) -> Vec<Segment> {
        let (x0, y0, x1, y1) = (
            TEXT.x0 - margin,
            TEXT.y0 - margin,
            TEXT.x1 + margin,
            TEXT.y1 + margin,
        );
        vec![
            Segment::MoveTo(pt(x0, y0)),
            Segment::LineTo(pt(x1, y0)),
            Segment::LineTo(pt(x1, y1)),
            Segment::LineTo(pt(x0, y1)),
            Segment::Close,
        ]
    }

    #[test]
    fn a_frame_paints_its_band_and_not_the_box_it_frames() {
        // A frame round the text, its band clear of it, its corners mitered.
        assert!(!solid(1.0, 0, 0, 10.0).may_paint(&frame(5.0), TEXT));
        // The planted wrong truths: the band reaching into the text, from a wider line or a nearer frame.
        assert!(solid(12.0, 0, 0, 10.0).may_paint(&frame(5.0), TEXT));
        assert!(solid(1.0, 0, 0, 10.0).may_paint(&frame(0.4), TEXT));
        // A frame along the text's edge with no width: it paints a line, sharing no area with the box.
        assert!(!solid(0.0, 0, 0, 10.0).may_paint(&frame(0.0), TEXT));
        // A frame's corners mitered square reach `√2` half widths, not the limit's 10: a line 5.8 wide (half 2.9), 3
        // round the text, clears it at every corner however it is joined.
        for join in [0, 1, 2] {
            assert!(!solid(5.8, 0, join, 10.0).may_paint(&frame(3.0), TEXT));
        }
    }

    #[test]
    fn a_line_paints_its_band_across_it_not_round_it() {
        // A thick line beside the text, 2 short of it: its band, half 3 wide, reaches over it along its length.
        let beside = line(pt(50.0, 108.0), pt(98.0, 108.0));
        assert!(!solid(6.0, 0, 0, 10.0).may_paint(&beside, TEXT));
        let above = line(pt(150.0, 50.0), pt(150.0, 98.0));
        assert!(!solid(4.0, 0, 0, 10.0).may_paint(&above, TEXT));
        assert!(solid(4.0, 0, 0, 10.0).may_paint(&line(pt(150.0, 50.0), pt(150.0, 101.0)), TEXT));
        // Diagonal, passing 2 from the text's corner: a band half 1 wide clears it, half 3 wide does not.
        let diagonal = line(
            pt(70.0, 70.0),
            pt(100.0 - 2.0f64.sqrt(), 100.0 - 2.0f64.sqrt()),
        )
        .into_iter()
        .chain([Segment::LineTo(pt(140.0, 60.0))])
        .collect::<Vec<_>>();
        assert!(!solid(2.0, 0, 1, 10.0).may_paint(&diagonal, TEXT));
        assert!(solid(6.0, 0, 1, 10.0).may_paint(&diagonal, TEXT));
    }

    #[test]
    fn a_square_or_round_cap_reaches_past_the_end_and_a_butt_cap_does_not() {
        // A line ending 2 short of the text, 6 wide: half a width (3) past its end is the cap's.
        let short = line(pt(50.0, 105.0), pt(98.0, 105.0));
        assert!(!solid(6.0, 0, 0, 10.0).may_paint(&short, TEXT));
        assert!(solid(6.0, 2, 0, 10.0).may_paint(&short, TEXT));
        assert!(solid(6.0, 1, 0, 10.0).may_paint(&short, TEXT));
        // A dot (a subpath of one point) paints its round cap alone.
        let dot = line(pt(98.0, 105.0), pt(98.0, 105.0));
        assert!(solid(6.0, 1, 0, 10.0).may_paint(&dot, TEXT));
        assert!(!solid(6.0, 0, 0, 10.0).may_paint(&dot, TEXT));
    }

    #[test]
    fn a_miter_reaches_as_far_as_its_limit_and_a_bevel_its_band() {
        // A sharp corner 4 below the text, pointing at it; 4 wide (half 2): its miter's tip may reach 2 * limit
        // (here it does: 2 / sin 12.3 deg, 9.4).
        let corner = vec![
            Segment::MoveTo(pt(140.0, 160.0)),
            Segment::LineTo(pt(150.0, 114.0)),
            Segment::LineTo(pt(160.0, 160.0)),
        ];
        assert!(solid(4.0, 0, 0, 10.0).may_paint(&corner, TEXT));
        assert!(!solid(4.0, 0, 2, 10.0).may_paint(&corner, TEXT));
        assert!(!solid(4.0, 0, 1, 10.0).may_paint(&corner, TEXT));
        assert!(!solid(4.0, 0, 0, 1.0).may_paint(&corner, TEXT)); // a limit of 1 bevels every corner
    }

    #[test]
    fn a_dash_caps_its_own_ends() {
        // A dashed line along the text, 2 below it, 3 wide: across a straight piece its dashes' caps add nothing.
        let along = line(pt(50.0, 112.0), pt(250.0, 112.0));
        let dashed = |cap| stroke(3.0, cap, 0, 10.0, &[4.0, 4.0], Matrix::IDENTITY);
        assert!(!dashed(2).may_paint(&along, TEXT));
        assert!(stroke(5.0, 0, 0, 10.0, &[4.0, 4.0], Matrix::IDENTITY).may_paint(&along, TEXT));
        // A line turning a corner 2 short of the text, its join round (half 1.5): solid, its square caps are at its
        // far ends; dashed, a dash may end at the corner, its square cap's corner `√2` half widths (2.12) from it.
        let turning = vec![
            Segment::MoveTo(pt(98.0, 60.0)),
            Segment::LineTo(pt(98.0, 105.0)),
            Segment::LineTo(pt(20.0, 105.0)),
        ];
        assert!(!stroke(3.0, 2, 1, 10.0, &[], Matrix::IDENTITY).may_paint(&turning, TEXT));
        assert!(stroke(3.0, 2, 1, 10.0, &[4.0, 4.0], Matrix::IDENTITY).may_paint(&turning, TEXT));
        // From a curve, the bound counts a square cap's corner every way: the bound, not the paint.
        let arc = vec![
            Segment::MoveTo(pt(50.0, 112.0)),
            Segment::CurveTo(pt(100.0, 112.0), pt(200.0, 112.0), pt(250.0, 112.0)),
        ];
        assert!(!dashed(0).may_paint(&arc, TEXT));
        assert!(dashed(2).may_paint(&arc, TEXT));
    }

    #[test]
    fn a_curve_is_bounded_by_halving_it_not_by_its_control_points_box() {
        // A circle of radius 80 round the text's middle: its control points' boxes hold the text, its band does not.
        let (cx, cy, r) = (150.0, 105.0, 80.0);
        let k = r * 0.552_284_75;
        let circle = vec![
            Segment::MoveTo(pt(cx + r, cy)),
            Segment::CurveTo(pt(cx + r, cy + k), pt(cx + k, cy + r), pt(cx, cy + r)),
            Segment::CurveTo(pt(cx - k, cy + r), pt(cx - r, cy + k), pt(cx - r, cy)),
            Segment::CurveTo(pt(cx - r, cy - k), pt(cx - k, cy - r), pt(cx, cy - r)),
            Segment::CurveTo(pt(cx + k, cy - r), pt(cx + r, cy - k), pt(cx + r, cy)),
            Segment::Close,
        ];
        assert!(!solid(2.0, 0, 0, 10.0).may_paint(&circle, TEXT));
        // The planted wrong truth: a circle whose band crosses the text's corners (they lie 50.25 from its middle).
        let near: Vec<Segment> = circle
            .iter()
            .map(|s| s.mapped(|p| pt(cx + (p.x - cx) * 50.0 / r, cy + (p.y - cy) * 50.0 / r)))
            .collect();
        assert!(solid(2.0, 0, 0, 10.0).may_paint(&near, TEXT));
        // A hairline circle through the text paints in it.
        assert!(solid(0.0, 0, 0, 10.0).may_paint(&near, TEXT));
    }

    #[test]
    fn a_matrix_stretching_one_way_widens_the_band_that_way() {
        // Width 1 in user space under x10, y0.1: the page space width (scaled by 1) is 1, but across a vertical line
        // the band is 10 wide.
        let squash = Matrix([10.0, 0.0, 0.0, 0.1, 0.0, 0.0]);
        let upright = line(pt(96.0, 50.0), pt(96.0, 150.0));
        let thin = stroke(1.0, 0, 0, 10.0, &[], squash);
        assert!((thin.half_width() - 5.0).abs() < 1e-9);
        assert!(thin.may_paint(&upright, TEXT));
        assert!(!solid(1.0, 0, 0, 10.0).may_paint(&upright, TEXT));
        // Along a horizontal line the same stroke is a tenth as wide (0.1): 0.5 below the text, it clears it.
        let level = line(pt(50.0, 110.5), pt(250.0, 110.5));
        assert!(!thin.may_paint(&level, TEXT));
        assert!(solid(1.2, 0, 0, 10.0).may_paint(&level, TEXT));
        // A skewing matrix slants the band's ends: a vertical line's band under x' = x + y, half 2 in user space,
        // leans 1.41 along it, so a line ending 1 short of the text reaches it.
        let shear = Matrix([1.0, 0.0, 1.0, 1.0, 0.0, 0.0]);
        let end = line(pt(150.0, 60.0), pt(150.0, 99.0));
        assert!(stroke(4.0, 0, 0, 10.0, &[], shear).may_paint(&end, TEXT));
        assert!(!solid(4.0, 0, 0, 10.0).may_paint(&end, TEXT));
        // A matrix with no area: its width is unknown, it may paint anything.
        let flat = stroke(0.0, 0, 0, 10.0, &[], Matrix([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]));
        assert!(flat.may_paint(&line(pt(0.0, 0.0), pt(1.0, 0.0)), TEXT));
    }

    #[test]
    fn reach_is_the_farthest_cap_or_join() {
        assert!((solid(2.0, 0, 1, 10.0).reach() - 1.0).abs() < 1e-12);
        assert!((solid(2.0, 2, 1, 10.0).reach() - std::f64::consts::SQRT_2).abs() < 1e-12);
        assert!((solid(2.0, 0, 0, 10.0).reach() - 10.0).abs() < 1e-12);
        assert!((solid(2.0, 0, 0, 0.5).reach() - 1.0).abs() < 1e-12);
    }
}
