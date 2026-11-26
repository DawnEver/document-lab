//! Painted paths to subpaths: each a polyline in page space, the unit a page's drawing is read in.
//!
//! A subpath runs from one `MoveTo` to the next; its straight segments give their ends, and each cubic Bezier
//! curve is flattened to points evenly spaced in its parameter, its end the last. A subpath of a lone point (a
//! `MoveTo` with nothing drawn from it) draws nothing and is left out.

use crate::geometry::Point;
use crate::interpret::Segment;

/// One subpath of a painted path.
#[derive(Debug, Clone, PartialEq)]
pub struct Subpath {
    /// Its points in drawing order, each curve flattened; two or more.
    pub points: Vec<Point>,
    /// Whether it is closed back to its start (`h`, a rectangle).
    pub closed: bool,
}

/// The subpaths `segments` draw, each curve flattened to `samples` points (at least one: its end).
#[must_use]
pub fn subpaths(segments: &[Segment], samples: usize) -> Vec<Subpath> {
    let samples = samples.max(1);
    let mut out = Vec::new();
    let mut open = Subpath {
        points: Vec::new(),
        closed: false,
    };
    let mut last = Point { x: 0.0, y: 0.0 };
    for segment in segments {
        match *segment {
            Segment::MoveTo(p) => {
                let done = std::mem::replace(
                    &mut open,
                    Subpath {
                        points: vec![p],
                        closed: false,
                    },
                );
                if done.points.len() > 1 {
                    out.push(done);
                }
                last = p;
            }
            Segment::LineTo(p) => {
                open.points.push(p);
                last = p;
            }
            Segment::CurveTo(c1, c2, p) => {
                open.points.extend((1..=samples).map(|i| {
                    #[expect(clippy::cast_precision_loss, reason = "a sample count is small")]
                    let t = i as f64 / samples as f64;
                    cubic([last, c1, c2, p], t)
                }));
                last = p;
            }
            Segment::Close => open.closed = true,
        }
    }
    if open.points.len() > 1 {
        out.push(open);
    }
    out
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

    #[test]
    fn a_curve_is_flattened_to_its_samples_ending_on_its_end() {
        let segments = [
            Segment::MoveTo(pt(10.0, 10.0)),
            Segment::CurveTo(pt(20.0, 0.0), pt(30.0, 0.0), pt(40.0, 10.0)),
        ];
        let [path] = subpaths(&segments, 4).try_into().unwrap();
        assert_eq!(path.points.len(), 5);
        assert_eq!(path.points[4], pt(40.0, 10.0));
        // Halfway along a symmetric curve: the middle of its span, three quarters of the way to its controls.
        assert_eq!(path.points[2], pt(25.0, 2.5));
        assert!(!path.closed);
    }

    #[test]
    fn no_sample_still_reaches_the_end() {
        let segments = [
            Segment::MoveTo(pt(0.0, 0.0)),
            Segment::CurveTo(pt(1.0, 1.0), pt(2.0, 1.0), pt(3.0, 0.0)),
        ];
        let [path] = subpaths(&segments, 0).try_into().unwrap();
        assert_eq!(path.points, [pt(0.0, 0.0), pt(3.0, 0.0)]);
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
        let [square, line] = subpaths(&segments, 8).try_into().unwrap();
        assert_eq!(
            (square.points.len(), square.closed),
            (3, true),
            "{square:?}"
        );
        assert_eq!(line.points, [pt(7.0, 7.0), pt(8.0, 7.0)]);
        assert!(!line.closed);
    }
}
