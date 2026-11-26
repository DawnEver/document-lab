//! Open every PDF under the paths given and resolve each of its objects; one line per file, then a summary.
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

fn main() {
    let mut files = Vec::new();
    for arg in std::env::args_os().skip(1) {
        pdfs(Path::new(&arg), &mut files);
    }
    let (mut sound, mut repaired, mut refused, mut bad_objects) = (0, 0, 0, 0);
    for path in &files {
        let data = match std::fs::read(path) {
            Ok(d) => d,
            Err(e) => {
                println!("UNREADABLE {}: {e}", path.display());
                refused += 1;
                continue;
            }
        };
        match pdf::File::open(data, b"") {
            Err(e) => {
                println!("REFUSED {}: {e}", path.display());
                refused += 1;
            }
            Ok(file) => {
                let numbers = file.numbers();
                // An object fails when it does not parse, or is a stream whose filters do not decode.
                let failed: Vec<_> = numbers
                    .iter()
                    .filter_map(|&n| match file.object(n) {
                        Err(e) => Some((n, e)),
                        Ok(pdf::Object::Stream(s)) => file.decode(&s).err().map(|e| (n, e)),
                        Ok(_) => None,
                    })
                    .collect();
                bad_objects += failed.len();
                let repairs = file.repairs();
                if repairs.is_empty() && failed.is_empty() {
                    sound += 1;
                } else {
                    repaired += 1;
                    println!(
                        "REPAIRED {}: {} repair(s), {} of {} objects unreadable",
                        path.display(),
                        repairs.len(),
                        failed.len(),
                        numbers.len()
                    );
                    for (n, e) in failed.iter().take(3) {
                        println!("    object {n}: {e}");
                    }
                }
            }
        }
    }
    println!(
        "{} files: {sound} sound, {repaired} repaired, {refused} refused; {bad_objects} objects unreadable",
        files.len()
    );
}
