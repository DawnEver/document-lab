//! Points and affine matrices (ISO 32000-2 section 8.3): `[a b c d e f]` maps `(x, y)` to
//! `(a x + c y + e, b x + d y + f)`.

use std::cmp::Ordering;

/// A point.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Point {
    /// Horizontal.
    pub x: f64,
    /// Vertical.
    pub y: f64,
}

/// An affine matrix `[a b c d e f]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Matrix(pub [f64; 6]);

impl Matrix {
    /// The identity.
    pub const IDENTITY: Self = Self([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);

    /// `self` then `then`: the matrix mapping a point through `self` first.
    #[must_use]
    pub fn then(self, then: Self) -> Self {
        let [a1, b1, c1, d1, e1, f1] = self.0;
        let [a2, b2, c2, d2, e2, f2] = then.0;
        Self([
            a1 * a2 + b1 * c2,
            a1 * b2 + b1 * d2,
            c1 * a2 + d1 * c2,
            c1 * b2 + d1 * d2,
            e1 * a2 + f1 * c2 + e2,
            e1 * b2 + f1 * d2 + f2,
        ])
    }

    /// `p` mapped.
    #[must_use]
    pub fn apply(self, p: Point) -> Point {
        let m = self.0;
        Point {
            x: m[0] * p.x + m[2] * p.y + m[4],
            y: m[1] * p.x + m[3] * p.y + m[5],
        }
    }

    /// How a length scales under this matrix: the square root of its area scale.
    #[must_use]
    pub fn scale(self) -> f64 {
        let [a, b, c, d, ..] = self.0;
        (a * d - b * c).abs().sqrt()
    }

    /// The most this matrix stretches a length, whatever its direction: its largest singular value (`scale` when it
    /// scales evenly).
    #[must_use]
    pub fn stretch(self) -> f64 {
        let [a, b, c, d, ..] = self.0;
        let sum = a * a + b * b + c * c + d * d;
        let det = a * d - b * c;
        f64::midpoint(sum, (sum * sum - 4.0 * det * det).max(0.0).sqrt()).sqrt()
    }

    /// The matrix in a PDF array of six numbers, if `values` is one.
    #[must_use]
    pub fn from_numbers(values: &[f64]) -> Option<Self> {
        <[f64; 6]>::try_from(values).ok().map(Self)
    }
}

/// An axis-aligned box `[x0, y0, x1, y1]`, `x0 <= x1` and `y0 <= y1`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rect {
    /// Left.
    pub x0: f64,
    /// Top in page space (y down); bottom in PDF user space.
    pub y0: f64,
    /// Right.
    pub x1: f64,
    /// Bottom in page space; top in user space.
    pub y1: f64,
}

impl Rect {
    /// The box spanning two corners in any order.
    #[must_use]
    pub fn spanning(a: Point, b: Point) -> Self {
        Self {
            x0: a.x.min(b.x),
            y0: a.y.min(b.y),
            x1: a.x.max(b.x),
            y1: a.y.max(b.y),
        }
    }

    /// The smallest box holding every point (at least one).
    #[must_use]
    pub fn around(points: &[Point]) -> Option<Self> {
        let first = *points.first()?;
        Some(
            points
                .iter()
                .fold(Self::spanning(first, first), |r, p| Self {
                    x0: r.x0.min(p.x),
                    y0: r.y0.min(p.y),
                    x1: r.x1.max(p.x),
                    y1: r.y1.max(p.y),
                }),
        )
    }

    /// Width.
    #[must_use]
    pub fn width(self) -> f64 {
        self.x1 - self.x0
    }

    /// Height.
    #[must_use]
    pub fn height(self) -> f64 {
        self.y1 - self.y0
    }

    /// The part both boxes cover; an empty box (`x0 > x1` or `y0 > y1`) when they do not meet.
    #[must_use]
    pub fn intersect(self, other: Self) -> Self {
        Self {
            x0: self.x0.max(other.x0),
            y0: self.y0.max(other.y0),
            x1: self.x1.min(other.x1),
            y1: self.y1.min(other.y1),
        }
    }

    /// The smallest box covering both.
    #[must_use]
    pub fn union(self, other: Self) -> Self {
        Self {
            x0: self.x0.min(other.x0),
            y0: self.y0.min(other.y0),
            x1: self.x1.max(other.x1),
            y1: self.y1.max(other.y1),
        }
    }

    /// Whether the boxes share a point (touching edges do).
    #[must_use]
    pub fn meets(self, other: Self) -> bool {
        self.x0 <= other.x1 && other.x0 <= self.x1 && self.y0 <= other.y1 && other.y0 <= self.y1
    }

    /// Whether the boxes share some area (touching edges do not).
    #[must_use]
    pub fn overlaps(self, other: Self) -> bool {
        self.x0 < other.x1 && other.x0 < self.x1 && self.y0 < other.y1 && other.y0 < self.y1
    }

    /// The box grown by `by` on every side.
    #[must_use]
    pub fn grown(self, by: f64) -> Self {
        Self {
            x0: self.x0 - by,
            y0: self.y0 - by,
            x1: self.x1 + by,
            y1: self.y1 + by,
        }
    }

    /// Its corners, clockwise in page space from the top left.
    #[must_use]
    pub fn corners(self) -> [Point; 4] {
        [
            Point {
                x: self.x0,
                y: self.y0,
            },
            Point {
                x: self.x1,
                y: self.y0,
            },
            Point {
                x: self.x1,
                y: self.y1,
            },
            Point {
                x: self.x0,
                y: self.y1,
            },
        ]
    }
}

/// The side of the line from `a` to `b` that `c` lies on -- the sign of their cross product -- when the floating
/// point product settles it: within Shewchuk's first error bound of zero (Adaptive Precision Floating-Point
/// Arithmetic and Fast Robust Geometric Predicates, 1997: `ccwerrboundA`) rounding could flip it, and None says so.
#[must_use]
pub fn orientation(a: Point, b: Point, c: Point) -> Option<Ordering> {
    const HALF_ULP: f64 = f64::EPSILON / 2.0;
    const BOUND: f64 = (3.0 + 16.0 * HALF_ULP) * HALF_ULP;
    let left = (b.x - a.x) * (c.y - a.y);
    let right = (b.y - a.y) * (c.x - a.x);
    let det = left - right;
    let bound = BOUND * (left.abs() + right.abs());
    (det.abs() > bound || bound <= 0.0)
        .then(|| det.partial_cmp(&0.0))
        .flatten()
}

/// Whether the segment from `p` to `q` reaches inside `bbox` (along an edge of it does not): unless an axis or the
/// segment's normal parts them -- every corner of the box on one side of its line, or on it. A corner no exact sign
/// places counts as on both sides.
#[must_use]
pub fn crosses(p: Point, q: Point, bbox: Rect) -> bool {
    if !Rect::spanning(p, q).overlaps(bbox) {
        return false;
    }
    let corners = bbox.corners().map(|c| orientation(p, q, c));
    let none_on = |side: Ordering| corners.iter().all(|c| c.is_some_and(|c| c != side));
    !(none_on(Ordering::Less) || none_on(Ordering::Greater))
}

/// Whether the segment from `p` to `q` meets `bbox`, its edges counted (touching does): unless an axis or the
/// segment's normal parts them -- every corner of the box strictly on one side of its line. A corner no exact sign
/// places counts as on the line.
#[must_use]
pub fn touches(p: Point, q: Point, bbox: Rect) -> bool {
    if !Rect::spanning(p, q).meets(bbox) {
        return false;
    }
    let corners = bbox.corners().map(|c| orientation(p, q, c));
    let all = |side: Ordering| corners.iter().all(|c| *c == Some(side));
    !(all(Ordering::Less) || all(Ordering::Greater))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn then_applies_the_first_matrix_first() {
        let scale = Matrix([2.0, 0.0, 0.0, 2.0, 0.0, 0.0]);
        let shift = Matrix([1.0, 0.0, 0.0, 1.0, 10.0, 0.0]);
        assert_eq!(
            scale.then(shift).apply(Point { x: 1.0, y: 1.0 }),
            Point { x: 12.0, y: 2.0 }
        );
        assert_eq!(
            shift.then(scale).apply(Point { x: 1.0, y: 1.0 }),
            Point { x: 22.0, y: 2.0 }
        );
        assert!((Matrix([3.0, 0.0, 0.0, 3.0, 5.0, 5.0]).scale() - 3.0).abs() < 1e-12);
    }

    #[test]
    fn stretch_is_the_most_a_length_grows() {
        // Even: the scale, turned or not.
        let (c, s) = (0.6, 0.8);
        assert!(
            (Matrix([3.0 * c, 3.0 * s, -3.0 * s, 3.0 * c, 1.0, 2.0]).stretch() - 3.0).abs() < 1e-12
        );
        // Uneven: the longer axis, though the area scale is 1.
        let squash = Matrix([10.0, 0.0, 0.0, 0.1, 0.0, 0.0]);
        assert!((squash.stretch() - 10.0).abs() < 1e-12 && (squash.scale() - 1.0).abs() < 1e-12);
        // A shear x' = x + y stretches the diagonal by the golden ratio.
        let golden = f64::midpoint(1.0, 5.0f64.sqrt());
        assert!((Matrix([1.0, 0.0, 1.0, 1.0, 0.0, 0.0]).stretch() - golden).abs() < 1e-12);
    }

    #[test]
    fn a_segment_touching_a_box_meets_it_and_reaches_inside_only_through_it() {
        let rect = Rect {
            x0: 0.0,
            y0: 0.0,
            x1: 10.0,
            y1: 10.0,
        };
        let pt = |x, y| Point { x, y };
        // Along an edge: touches, does not cross; the boxes meet, they do not overlap.
        assert!(touches(pt(-5.0, 0.0), pt(15.0, 0.0), rect));
        assert!(!crosses(pt(-5.0, 0.0), pt(15.0, 0.0), rect));
        let edge = Rect::spanning(pt(-5.0, 0.0), pt(15.0, 0.0));
        assert!(edge.meets(rect) && !edge.overlaps(rect));
        // Through it: both. Past its corner: neither.
        assert!(
            touches(pt(-5.0, 5.0), pt(15.0, 5.0), rect)
                && crosses(pt(-5.0, 5.0), pt(15.0, 5.0), rect)
        );
        assert!(!touches(pt(-5.0, 4.0), pt(4.0, -5.0), rect));
        assert!(!crosses(pt(-5.0, 4.0), pt(4.0, -5.0), rect));
    }
}

/// One piece of a path: in page space on the interpreter's events, in text space on a glyph's outline.
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

impl Segment {
    /// The segment with each of its points mapped by `map`.
    #[must_use]
    pub fn mapped(self, map: impl Fn(Point) -> Point) -> Self {
        match self {
            Self::MoveTo(p) => Self::MoveTo(map(p)),
            Self::LineTo(p) => Self::LineTo(map(p)),
            Self::CurveTo(a, b, p) => Self::CurveTo(map(a), map(b), map(p)),
            Self::Close => Self::Close,
        }
    }
}
