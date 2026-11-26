//! Where the crate's reading time goes: for every PDF under the paths given, the wall time of opening it, of
//! interpreting its pages, of setting their text into spans and of flattening their paths; per file and in total,
//! the slowest files first.
//!
//! `cargo run --release --example timing -- [--samples N] <file-or-directory>...` -- a measurement against real
//! documents kept outside the repository; nothing is written.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use pdf::paths::Polylines;

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

/// The time each stage took on one file.
#[derive(Default, Clone, Copy)]
struct Stages {
    open: Duration,
    interpret: Duration,
    spans: Duration,
    flatten: Duration,
    pages: usize,
    points: usize,
}

impl Stages {
    fn total(&self) -> Duration {
        self.open + self.interpret + self.spans + self.flatten
    }

    fn add(&mut self, other: &Self) {
        self.open += other.open;
        self.interpret += other.interpret;
        self.spans += other.spans;
        self.flatten += other.flatten;
        self.pages += other.pages;
        self.points += other.points;
    }

    fn line(&self, name: &str) -> String {
        let ms = |d: Duration| d.as_secs_f64() * 1e3;
        format!(
            "{:9.1} ms  open {:8.1}  interpret {:8.1}  spans {:7.1}  flatten {:7.1}  {:5} pages {:9} points  {name}",
            ms(self.total()),
            ms(self.open),
            ms(self.interpret),
            ms(self.spans),
            ms(self.flatten),
            self.pages,
            self.points
        )
    }
}

fn time(path: &Path, samples: usize) -> Option<Stages> {
    let data = std::fs::read(path).ok()?;
    let mut s = Stages::default();
    let start = Instant::now();
    let file = pdf::File::open(data, b"").ok()?;
    let pages = file.pages().ok()?;
    s.open = start.elapsed();
    for page in &pages {
        let start = Instant::now();
        let Ok(content) = file.interpret(page) else {
            continue;
        };
        s.interpret += start.elapsed();
        let start = Instant::now();
        let spans = pdf::spans::spans(&content.events);
        s.spans += start.elapsed();
        let start = Instant::now();
        let mut lines = Polylines::new();
        for event in &content.events {
            if let pdf::Event::Path(path) = event {
                lines.add(&path.segments, samples);
            }
        }
        s.flatten += start.elapsed();
        s.pages += 1;
        s.points += lines.points.len();
        drop(spans);
    }
    Some(s)
}

fn main() {
    let mut args = std::env::args_os().skip(1).peekable();
    let mut samples = 8;
    if args.peek().is_some_and(|a| a == "--samples") {
        args.next();
        samples = args
            .next()
            .and_then(|n| n.to_str()?.parse().ok())
            .expect("--samples takes a count");
    }
    let mut files = Vec::new();
    for arg in args {
        pdfs(Path::new(&arg), &mut files);
    }
    let mut timed: Vec<(PathBuf, Stages)> = files
        .into_iter()
        .filter_map(|f| time(&f, samples).map(|s| (f, s)))
        .collect();
    timed.sort_by_key(|(_, s)| std::cmp::Reverse(s.total()));
    let mut all = Stages::default();
    for (_, s) in &timed {
        all.add(s);
    }
    for (file, s) in timed.iter().take(15) {
        let name = file
            .file_name()
            .map_or_else(String::new, |n| n.to_string_lossy().into_owned());
        println!("{}", s.line(&name));
    }
    println!("{}", all.line(&format!("TOTAL of {} files", timed.len())));
}
