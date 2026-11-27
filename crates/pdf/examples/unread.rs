//! Count, by font, the glyphs whose text no source gives (no `ToUnicode`, encoding or font program entry).
//!
//! `cargo run --release --example unread -- <file-or-directory>...`; prints the 40 fonts with the most.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

fn pdfs(path: &Path, out: &mut Vec<PathBuf>) {
    if path.is_dir() {
        for entry in std::fs::read_dir(path).into_iter().flatten().flatten() {
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
    for arg in std::env::args_os().skip(1) {
        pdfs(Path::new(&arg), &mut files);
    }
    let mut counts: HashMap<String, usize> = HashMap::new();
    for path in files {
        let Ok(file) = pdf::File::open(std::fs::read(&path).unwrap_or_default(), b"") else {
            continue;
        };
        for page in file.pages().unwrap_or_default() {
            for event in file
                .interpret(&page, false)
                .map_or_else(|_| Vec::new(), |c| c.events)
            {
                if let pdf::Event::Text(text) = event {
                    let unread = text.glyphs.iter().filter(|g| g.text.is_none()).count();
                    if unread > 0 {
                        *counts
                            .entry(format!(
                                "{} {}",
                                text.font.info.subtype, text.font.info.name
                            ))
                            .or_default() += unread;
                    }
                }
            }
        }
    }
    let mut ranked: Vec<_> = counts.into_iter().collect();
    ranked.sort_by_key(|r| std::cmp::Reverse(r.1));
    for (font, n) in ranked.iter().take(40) {
        println!("{n:7} {font}");
    }
}
