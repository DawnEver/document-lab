//! Render every page of every PDF under the paths given at 150 dpi (no text, annotations drawn) and report the time
//! a page takes, interpretation included, with the slowest pages and what was not drawn.
//!
//! `cargo run --release -p render --example time -- <file-or-directory>...` -- a measure on real documents kept
//! outside the repository; nothing is written.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

fn pdfs(path: &Path, out: &mut Vec<PathBuf>) {
    if path.is_dir() {
        let Ok(entries) = std::fs::read_dir(path) else {
            return;
        };
        for entry in entries.flatten() {
            pdfs(&entry.path(), out);
        }
    } else if path
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("pdf"))
    {
        out.push(path.to_path_buf());
    }
}

fn main() {
    let mut files = Vec::new();
    for arg in std::env::args().skip(1) {
        pdfs(Path::new(&arg), &mut files);
    }
    files.sort();
    let options = render::Options {
        region: None,
        scale: 150.0 / 72.0,
        text: false,
        annotations: true,
        alpha: false,
        jpx: render::image::Jpx::default(),
    };
    let (mut pages, mut total, mut failed) = (0usize, 0.0f64, 0usize);
    let mut slowest: Vec<(f64, String)> = Vec::new();
    let mut undrawn: BTreeMap<String, usize> = BTreeMap::new();
    for path in &files {
        let Ok(data) = std::fs::read(path) else {
            continue;
        };
        let Ok(file) = pdf::File::open(data, b"") else {
            continue;
        };
        let Ok(all) = file.pages() else {
            continue;
        };
        for page in &all {
            let start = Instant::now();
            match render::render(&file, page, &options) {
                Ok(raster) => {
                    let ms = start.elapsed().as_secs_f64() * 1e3;
                    pages += 1;
                    total += ms;
                    slowest.push((ms, format!("{} p{}", path.display(), page.number)));
                    for (what, n) in raster.undrawn {
                        *undrawn.entry(what).or_default() += n;
                    }
                }
                Err(e) => {
                    failed += 1;
                    println!("{} p{}: {e}", path.display(), page.number);
                }
            }
        }
    }
    slowest.sort_by(|a, b| b.0.total_cmp(&a.0));
    for (ms, page) in slowest.iter().take(5) {
        println!("{ms:8.1} ms  {page}");
    }
    #[expect(clippy::cast_precision_loss, reason = "a page count")]
    let mean = total / pages.max(1) as f64;
    println!(
        "{} files, {pages} pages rendered ({failed} refused): {mean:.1} ms a page",
        files.len()
    );
    for (what, n) in &undrawn {
        println!("undrawn: {what} x{n}");
    }
}
