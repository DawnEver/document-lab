//! An image's placement: the quarter turn and mirrors its matrix shows its samples with, or none when it is skewed.

use pdf::geometry::{Matrix, Point};
use pdf::image::Placement;

/// Samples across and down of the image placed.
const WIDTH: usize = 3;
const HEIGHT: usize = 2;

/// Each sample's index (row * WIDTH + column) as the page shows it under `matrix`: rows down the page, top first,
/// each left to right -- found by where the matrix puts each sample's center.
fn shown(matrix: &Matrix) -> Vec<Vec<usize>> {
    #[expect(clippy::cast_precision_loss, reason = "a few samples")]
    let at: Vec<(i64, i64, usize)> = (0..HEIGHT * WIDTH)
        .map(|k| {
            let (row, column) = (k / WIDTH, k % WIDTH);
            let at = matrix.apply(Point {
                x: (column as f64 + 0.5) / WIDTH as f64,
                y: 1.0 - (row as f64 + 0.5) / HEIGHT as f64,
            });
            #[expect(
                clippy::cast_possible_truncation,
                reason = "a small coordinate, rounded"
            )]
            let key = |z: f64| (z * 1000.0).round() as i64;
            (key(at.x), key(at.y), k)
        })
        .collect();
    let mut ys: Vec<i64> = at.iter().map(|p| p.1).collect();
    ys.sort_unstable();
    ys.dedup();
    ys.iter()
        .map(|&y| {
            let mut row: Vec<(i64, usize)> =
                at.iter().filter(|p| p.1 == y).map(|p| (p.0, p.2)).collect();
            row.sort_unstable();
            row.into_iter().map(|(_, k)| k).collect()
        })
        .collect()
}

/// The samples as `placement` says the page shows them.
fn said(placement: Placement) -> Vec<Vec<usize>> {
    let mut rows: Vec<Vec<usize>> = if placement.turned {
        (0..WIDTH)
            .map(|j| (0..HEIGHT).map(|i| i * WIDTH + j).collect())
            .collect()
    } else {
        (0..HEIGHT)
            .map(|i| (0..WIDTH).map(|j| i * WIDTH + j).collect())
            .collect()
    };
    if placement.rows_reversed {
        rows.reverse();
    }
    if placement.columns_reversed {
        for row in &mut rows {
            row.reverse();
        }
    }
    rows
}

#[test]
fn every_quarter_turn_and_mirror_is_told_as_the_page_shows_it() {
    // page space, y down: upright, mirrored, half a turn, flipped; then each of those turned a quarter
    let placements = [
        [3.0, 0.0, 0.0, -2.0, 0.0, 2.0],
        [-3.0, 0.0, 0.0, -2.0, 3.0, 2.0],
        [-3.0, 0.0, 0.0, 2.0, 3.0, 0.0],
        [3.0, 0.0, 0.0, 2.0, 0.0, 0.0],
        [0.0, 3.0, -2.0, 0.0, 2.0, 0.0],
        [0.0, 3.0, 2.0, 0.0, 0.0, 0.0],
        [0.0, -3.0, 2.0, 0.0, 0.0, 3.0],
        [0.0, -3.0, -2.0, 0.0, 2.0, 3.0],
    ];
    let mut told = Vec::new();
    for m in placements {
        let matrix = Matrix(m);
        let placement = Placement::of(&matrix).unwrap_or_else(|| panic!("{m:?} is square"));
        assert_eq!(said(placement), shown(&matrix), "{m:?}: {placement:?}");
        told.push(placement);
    }
    told.sort_by_key(|p| (p.turned, p.rows_reversed, p.columns_reversed));
    told.dedup();
    assert_eq!(told.len(), 8, "each placement is its own class");
}

#[test]
fn a_skewed_placement_is_none_and_a_rounding_error_is_square() {
    let (sin, cos) = 30f64.to_radians().sin_cos();
    assert_eq!(
        Placement::of(&Matrix([
            3.0 * cos,
            3.0 * sin,
            -2.0 * sin,
            2.0 * cos,
            0.0,
            0.0
        ])),
        None
    );
    assert_eq!(
        Placement::of(&Matrix([3.0, 0.5, 0.0, -2.0, 0.0, 2.0])),
        None
    ); // sheared
    let nearly = Placement::of(&Matrix([3.0, 1e-9, -1e-9, -2.0, 0.0, 2.0]));
    assert_eq!(
        nearly,
        Some(Placement {
            turned: false,
            rows_reversed: false,
            columns_reversed: false
        })
    );
}
