//! Open every PDF under the paths given, resolve and decode each object, and interpret each page with its
//! annotations; one line per file that needed a repair or failed somewhere, then a summary.
//!
//! `cargo run --release --example scan -- <file-or-directory>...` -- a check against real documents kept outside
//! the repository; nothing is written.

use std::path::{Path, PathBuf};

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

#[derive(Default)]
struct Totals {
    files: usize,
    sound: usize,
    refused: usize,
    bad_objects: usize,
    pages: usize,
    bad_pages: usize,
    warnings: usize,
    paths: usize,
    images: usize,
    shadings: usize,
    glyphs: usize,
    unmapped: usize,
}

fn scan(path: &Path, t: &mut Totals) {
    t.files += 1;
    let opened = std::fs::read(path)
        .map_err(|e| e.to_string())
        .and_then(|d| pdf::File::open(d, b"").map_err(|e| e.to_string()));
    let file = match opened {
        Ok(f) => f,
        Err(e) => {
            println!("REFUSED {}: {e}", path.display());
            t.refused += 1;
            return;
        }
    };
    let mut problems = Vec::new();
    for n in file.numbers() {
        let failed = match file.object(n) {
            Err(e) => Some(e),
            Ok(pdf::Object::Stream(s)) => file.decode(&s).err(),
            Ok(_) => None,
        };
        if let Some(e) = failed {
            t.bad_objects += 1;
            problems.push(format!("object {n}: {e}"));
        }
    }
    match file.pages() {
        Err(e) => problems.push(format!("pages: {e}")),
        Ok(pages) => {
            for page in &pages {
                t.pages += 1;
                match file.interpret(page, true) {
                    Err(e) => {
                        t.bad_pages += 1;
                        problems.push(format!("page {}: {e}", page.number));
                    }
                    Ok(content) => {
                        t.warnings += content.warnings.len();
                        for w in content.warnings.iter().take(2) {
                            problems.push(format!("page {} warning: {}", page.number, w.why));
                        }
                        for e in &content.events {
                            match e {
                                pdf::Event::Path(_) => t.paths += 1,
                                pdf::Event::Image(_) => t.images += 1,
                                pdf::Event::Shading(_) => t.shadings += 1,
                                pdf::Event::Text(text) => {
                                    t.glyphs += text.glyphs.len();
                                    t.unmapped +=
                                        text.glyphs.iter().filter(|g| g.text.is_none()).count();
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    problems.extend(file.repairs());
    if problems.is_empty() {
        t.sound += 1;
    } else {
        println!("{}: {} problem(s)", path.display(), problems.len());
        for p in problems.iter().take(4) {
            println!("    {p}");
        }
    }
}

fn main() {
    let mut files = Vec::new();
    for arg in std::env::args_os().skip(1) {
        pdfs(Path::new(&arg), &mut files);
    }
    let mut t = Totals::default();
    for path in &files {
        scan(path, &mut t);
    }
    println!(
        "{} files: {} sound, {} refused, {} unreadable objects; {} pages, {} failed, {} warnings; {} paths, {} images, {} shadings, {} glyphs ({} without text)",
        t.files,
        t.sound,
        t.refused,
        t.bad_objects,
        t.pages,
        t.bad_pages,
        t.warnings,
        t.paths,
        t.images,
        t.shadings,
        t.glyphs,
        t.unmapped
    );
}
