//! Painted paths to subpaths: each a polyline in page space, the unit a page's drawing is read in.
//!
//! A subpath runs from one `MoveTo` to the next; its straight segments give their ends, and each cubic Bezier
//! curve is flattened to points evenly spaced in its parameter, its end the last. A subpath of a lone point (a
//! `MoveTo` with nothing drawn from it) draws nothing and is left out.
//!
//! Subpaths are kept as [`Polylines`]: every point in one run, each subpath a range of it -- a page's drawing is
//! hundreds of thousands of subpaths, and one buffer hands over whole (`document_lab._pdf`).

use crate::geometry::Point;
use crate::interpret::Segment;

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

    /// Append the subpaths `segments` draw, each curve flattened to `samples` points (at least one: its end).
    pub fn add(&mut self, segments: &[Segment], samples: usize) {
        let samples = samples.max(1);
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
                    self.points.extend((1..=samples).map(|i| {
                        #[expect(clippy::cast_precision_loss, reason = "a sample count is small")]
                        let t = i as f64 / samples as f64;
                        cubic([last, c1, c2, p], t)
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

    fn flattened(segments: &[Segment], samples: usize) -> Polylines {
        let mut lines = Polylines::new();
        lines.add(segments, samples);
        lines
    }

    #[test]
    fn a_curve_is_flattened_to_its_samples_ending_on_its_end() {
        let segments = [
            Segment::MoveTo(pt(10.0, 10.0)),
            Segment::CurveTo(pt(20.0, 0.0), pt(30.0, 0.0), pt(40.0, 10.0)),
        ];
        let lines = flattened(&segments, 4);
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
        assert_eq!(flattened(&segments, 0).points, [pt(0.0, 0.0), pt(3.0, 0.0)]);
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
        let lines = flattened(&segments, 8);
        assert_eq!(lines.bounds, [0, 3, 5]);
        assert_eq!(lines.closed, [true, false]);
        assert_eq!(lines.subpath(1), [pt(7.0, 7.0), pt(8.0, 7.0)]);
    }

    #[test]
    fn paths_added_one_after_another_share_one_run() {
        let mut lines = Polylines::new();
        lines.add(
            &[Segment::MoveTo(pt(0.0, 0.0)), Segment::LineTo(pt(1.0, 0.0))],
            8,
        );
        lines.add(&[Segment::MoveTo(pt(9.0, 9.0))], 8);
        lines.add(
            &[
                Segment::MoveTo(pt(2.0, 2.0)),
                Segment::LineTo(pt(3.0, 2.0)),
                Segment::LineTo(pt(3.0, 3.0)),
            ],
            8,
        );
        assert_eq!(lines.bounds, [0, 2, 5]);
        assert_eq!(lines.points.len(), 5, "the lone point left no trace");
        assert_eq!(lines.subpath(1)[0], pt(2.0, 2.0));
    }
}
