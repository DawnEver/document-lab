//! Points and affine matrices (ISO 32000-2 section 8.3): `[a b c d e f]` maps `(x, y)` to
//! `(a x + c y + e, b x + d y + f)`.

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
