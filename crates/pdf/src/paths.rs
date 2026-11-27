//! Painted paths to subpaths: each a polyline in page space, the unit a page's drawing is read in.
//!
//! A subpath runs from one `MoveTo` to the next; its straight segments give their ends, and each cubic Bezier
//! curve is flattened to points evenly spaced in its parameter, its end the last -- as many as [`Flattening`]
//! says: a count for every curve, or as many as keep the polyline within a distance of the curve. A subpath of a
//! lone point (a `MoveTo` with nothing drawn from it) draws nothing and is left out.
//!
//! Subpaths are kept as [`Polylines`]: every point in one run, each subpath a range of it -- a page's drawing is
//! hundreds of thousands of subpaths, and one buffer hands over whole (`document_lab._pdf`).

use crate::geometry::Point;
use crate::geometry::Segment;

/// The most segments one curve is flattened to: a bound on what absurd coordinates can cost, far past any
/// curve a page draws within a flatness of a hundredth of a point.
pub const MAX_SEGMENTS: usize = 4096;

/// How a curve is flattened.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Flattening {
    /// Into this many segments, whatever its size (at least one).
    Samples(usize),
    /// Into as few segments as keep every point of the polyline within this distance (page-space points) of the
    /// curve: a glyph's curve takes one or two, a page-wide arc dozens.
    Flatness(f64),
}

impl Flattening {
    /// The segments the cubic Bezier curve with control points `p` is flattened into.
    ///
    /// For a flatness `d`, Wang's formula: `n` segments even in the parameter keep the polyline within
    /// `3/4 L / n^2` of the curve, `L` the larger of its two second differences `|p0 - 2 p1 + p2|` and
    /// `|p1 - 2 p2 + p3|`; so `n = ceil(sqrt(3/4 L / d))`, at most [`MAX_SEGMENTS`].
    #[must_use]
    pub fn segments(self, p: [Point; 4]) -> usize {
        match self {
            Self::Samples(n) => n.max(1),
            Self::Flatness(d) => {
                let second = |a: Point, b: Point, c: Point| {
                    (a.x - 2.0 * b.x + c.x).hypot(a.y - 2.0 * b.y + c.y)
                };
                let bend = second(p[0], p[1], p[2]).max(second(p[1], p[2], p[3]));
                #[expect(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "a float casts saturating: NaN (a degenerate flatness) and below 0 to 0"
                )]
                let n = (0.75 * bend / d).sqrt().ceil() as usize;
                n.clamp(1, MAX_SEGMENTS)
            }
        }
    }
}

/// Subpaths, their points one run: subpath `i` is `points[bounds[i]..bounds[i + 1]]`.
#[derive(Debug, Clone, PartialEq)]
pub struct Polylines {
    /// Every subpath's points in drawing order, subpath after subpath, each curve flattened.
    pub points: Vec<Point>,
    /// Where each subpath starts in `points`, then where the last ends: one more than there are subpaths.
    pub bounds: Vec<usize>,
    /// Whether each subpath is closed back to its start (`h`, a rectangle).
    pub closed: Vec<bool>,
}

impl Default for Polylines {
    fn default() -> Self {
        Self {
            points: Vec::new(),
            bounds: vec![0],
            closed: Vec::new(),
        }
    }
}

impl Polylines {
    /// No subpath.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// How many subpaths.
    #[must_use]
    pub fn len(&self) -> usize {
        self.closed.len()
    }

    /// Whether there is none.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.closed.is_empty()
    }

    /// The points of subpath `i`: two or more.
    #[must_use]
    pub fn subpath(&self, i: usize) -> &[Point] {
        &self.points[self.bounds[i]..self.bounds[i + 1]]
    }

    /// Append the subpaths `segments` draw, each curve flattened as `flattening` says.
    pub fn add(&mut self, segments: &[Segment], flattening: Flattening) {
        let mut closed = false;
        let mut last = Point { x: 0.0, y: 0.0 };
        for segment in segments {
            match *segment {
                Segment::MoveTo(p) => {
                    self.end(closed);
                    closed = false;
                    self.points.push(p);
                }
                Segment::LineTo(p) => self.points.push(p),
                Segment::CurveTo(c1, c2, p) => {
                    let curve = [last, c1, c2, p];
                    let samples = flattening.segments(curve);
                    self.points.extend((1..=samples).map(|i| {
                        #[expect(clippy::cast_precision_loss, reason = "a sample count is small")]
                        let t = i as f64 / samples as f64;
                        cubic(curve, t)
                    }));
                }
                Segment::Close => closed = true,
            }
            if let Segment::MoveTo(p) | Segment::LineTo(p) | Segment::CurveTo(_, _, p) = *segment {
                last = p;
            }
        }
        self.end(closed);
    }

    /// End the subpath open since the last bound: kept when it has two points or more, else dropped.
    fn end(&mut self, closed: bool) {
        let start = self.bounds[self.bounds.len() - 1];
        if self.points.len() - start > 1 {
            self.bounds.push(self.points.len());
            self.closed.push(closed);
        } else {
            self.points.truncate(start);
        }
    }
}

/// The point at parameter `t` (0..=1) of the cubic Bezier curve with control points `p`.
fn cubic(p: [Point; 4], t: f64) -> Point {
    let u = 1.0 - t;
    let weights = [u * u * u, 3.0 * u * u * t, 3.0 * u * t * t, t * t * t];
    let at = |axis: fn(&Point) -> f64| {
        weights
            .iter()
            .zip(&p)
            .map(|(w, point)| w * axis(point))
            .sum()
    };
    Point {
        x: at(|q| q.x),
        y: at(|q| q.y),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pt(x: f64, y: f64) -> Point {
        Point { x, y }
    }

    fn flattened(segments: &[Segment], flattening: Flattening) -> Polylines {
        let mut lines = Polylines::new();
        lines.add(segments, flattening);
        lines
    }

    #[test]
    fn a_curve_is_flattened_to_its_samples_ending_on_its_end() {
        let segments = [
            Segment::MoveTo(pt(10.0, 10.0)),
            Segment::CurveTo(pt(20.0, 0.0), pt(30.0, 0.0), pt(40.0, 10.0)),
        ];
        let lines = flattened(&segments, Flattening::Samples(4));
        assert_eq!(lines.len(), 1);
        let path = lines.subpath(0);
        assert_eq!(path.len(), 5);
        assert_eq!(path[4], pt(40.0, 10.0));
        // Halfway along a symmetric curve: the middle of its span, three quarters of the way to its controls.
        assert_eq!(path[2], pt(25.0, 2.5));
        assert_eq!(lines.closed, [false]);
    }

    #[test]
    fn no_sample_still_reaches_the_end() {
        let segments = [
            Segment::MoveTo(pt(0.0, 0.0)),
            Segment::CurveTo(pt(1.0, 1.0), pt(2.0, 1.0), pt(3.0, 0.0)),
        ];
        assert_eq!(
            flattened(&segments, Flattening::Samples(0)).points,
            [pt(0.0, 0.0), pt(3.0, 0.0)]
        );
    }

    #[test]
    fn each_move_starts_a_subpath_and_a_lone_point_draws_none() {
        let segments = [
            Segment::MoveTo(pt(0.0, 0.0)),
            Segment::LineTo(pt(1.0, 0.0)),
            Segment::LineTo(pt(1.0, 1.0)),
            Segment::Close,
            Segment::MoveTo(pt(5.0, 5.0)),
            Segment::MoveTo(pt(7.0, 7.0)),
            Segment::LineTo(pt(8.0, 7.0)),
        ];
        let lines = flattened(&segments, Flattening::Samples(8));
        assert_eq!(lines.bounds, [0, 3, 5]);
        assert_eq!(lines.closed, [true, false]);
        assert_eq!(lines.subpath(1), [pt(7.0, 7.0), pt(8.0, 7.0)]);
    }

    #[test]
    fn paths_added_one_after_another_share_one_run() {
        let mut lines = Polylines::new();
        lines.add(
            &[Segment::MoveTo(pt(0.0, 0.0)), Segment::LineTo(pt(1.0, 0.0))],
            Flattening::Samples(8),
        );
        lines.add(&[Segment::MoveTo(pt(9.0, 9.0))], Flattening::Samples(8));
        lines.add(
            &[
                Segment::MoveTo(pt(2.0, 2.0)),
                Segment::LineTo(pt(3.0, 2.0)),
                Segment::LineTo(pt(3.0, 3.0)),
            ],
            Flattening::Samples(8),
        );
        assert_eq!(lines.bounds, [0, 2, 5]);
        assert_eq!(lines.points.len(), 5, "the lone point left no trace");
        assert_eq!(lines.subpath(1)[0], pt(2.0, 2.0));
    }

    /// The farthest a point of the curve `p` lies from `polyline` (dense samples of the curve, each to its nearest
    /// segment).
    fn distance(p: [Point; 4], polyline: &[Point]) -> f64 {
        let to_segment = |q: Point, a: Point, b: Point| {
            let (dx, dy) = (b.x - a.x, b.y - a.y);
            let length = dx * dx + dy * dy;
            let t = if length > 0.0 {
                (((q.x - a.x) * dx + (q.y - a.y) * dy) / length).clamp(0.0, 1.0)
            } else {
                0.0
            };
            (q.x - a.x - t * dx).hypot(q.y - a.y - t * dy)
        };
        (0..=2000)
            .map(|i| cubic(p, f64::from(i) / 2000.0))
            .map(|q| {
                polyline
                    .windows(2)
                    .map(|w| to_segment(q, w[0], w[1]))
                    .fold(f64::INFINITY, f64::min)
            })
            .fold(0.0, f64::max)
    }

    #[test]
    fn a_flatness_keeps_the_polyline_that_close_to_the_curve_with_few_points() {
        // A quarter circle of radius 100 points, and the same shape a thousandth the size (a glyph's curve).
        let k = 0.552_284_75;
        for scale in [100.0, 0.1] {
            let curve = [
                pt(scale, 0.0),
                pt(scale, scale * k),
                pt(scale * k, scale),
                pt(0.0, scale),
            ];
            for flatness in [0.5, 0.1, 0.01] {
                let lines = flattened(
                    &[
                        Segment::MoveTo(curve[0]),
                        Segment::CurveTo(curve[1], curve[2], curve[3]),
                    ],
                    Flattening::Flatness(flatness),
                );
                let far = distance(curve, lines.subpath(0));
                assert!(far <= flatness, "{scale} {flatness}: {far}");
                if scale < 1.0 && flatness >= 0.1 {
                    assert_eq!(
                        lines.points.len(),
                        2,
                        "a curve flatter than asked is one segment"
                    );
                }
            }
        }
    }

    #[test]
    fn a_fixed_count_can_miss_a_large_curve_by_more_than_a_flatness_does() {
        let k = 0.552_284_75;
        let curve = [
            pt(300.0, 0.0),
            pt(300.0, 300.0 * k),
            pt(300.0 * k, 300.0),
            pt(0.0, 300.0),
        ];
        let draw = [
            Segment::MoveTo(curve[0]),
            Segment::CurveTo(curve[1], curve[2], curve[3]),
        ];
        let fixed = distance(curve, flattened(&draw, Flattening::Samples(8)).subpath(0));
        let tolerant = distance(
            curve,
            flattened(&draw, Flattening::Flatness(0.1)).subpath(0),
        );
        assert!(fixed > 0.5 && tolerant <= 0.1, "{fixed} {tolerant}");
    }

    #[test]
    fn a_degenerate_flatness_still_reaches_the_end() {
        let curve = [pt(0.0, 0.0), pt(1e300, 0.0), pt(-1e300, 1.0), pt(3.0, 0.0)];
        for flatness in [0.0, -1.0, f64::NAN] {
            let n = Flattening::Flatness(flatness).segments(curve);
            assert!((1..=MAX_SEGMENTS).contains(&n), "{flatness}: {n}");
        }
        let lines = flattened(
            &[
                Segment::MoveTo(curve[0]),
                Segment::CurveTo(curve[1], curve[2], curve[3]),
            ],
            Flattening::Flatness(f64::NAN),
        );
        assert_eq!(lines.points.last(), Some(&pt(3.0, 0.0)));
    }
}
