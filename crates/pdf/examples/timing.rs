//! Where the crate's reading time goes: for every PDF under the paths given, the wall time of opening it, of
//! interpreting its pages, of setting their text into spans and of flattening their paths; per file and in total,
//! the slowest files first. Of interpreting, the part spent decoding the pages' own content streams and lexing
//! them into operations is timed apart as well (a Form's content is not in it): the rest is running them.
//!
//! `cargo run --release --example timing -- [--samples N | --flatness D] <file-or-directory>...` (default: 8
//! samples a curve) -- a measurement against real
//! documents kept outside the repository; nothing is written.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use pdf::paths::{Flattening, Polylines};

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
    decode: Duration,
    lex: Duration,
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
        self.decode += other.decode;
        self.lex += other.lex;
        self.pages += other.pages;
        self.points += other.points;
    }

    fn line(&self, name: &str) -> String {
        let ms = |d: Duration| d.as_secs_f64() * 1e3;
        format!(
            "{:9.1} ms  open {:7.1}  interpret {:8.1} (decode {:7.1} lex {:7.1})  spans {:7.1}  flatten {:7.1}  {:5} pages {:9} points  {name}",
            ms(self.total()),
            ms(self.open),
            ms(self.interpret),
            ms(self.decode),
            ms(self.lex),
            ms(self.spans),
            ms(self.flatten),
            self.pages,
            self.points
        )
    }
}

fn time(path: &Path, flattening: Flattening) -> Option<Stages> {
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
        let data = contents(&file, page);
        s.decode += start.elapsed();
        let start = Instant::now();
        drop(pdf::content::operations(&data, |_| {}));
        s.lex += start.elapsed();
        let start = Instant::now();
        let spans = pdf::spans::spans(&content.events);
        s.spans += start.elapsed();
        let start = Instant::now();
        let mut lines = Polylines::new();
        for event in &content.events {
            if let pdf::Event::Path(path) = event {
                lines.add(&path.segments, flattening);
            }
        }
        s.flatten += start.elapsed();
        s.pages += 1;
        s.points += lines.points.len();
        drop(spans);
    }
    Some(s)
}

/// The page's own content streams, decoded and joined.
fn contents(file: &pdf::File, page: &pdf::Page) -> Vec<u8> {
    let decoded = |s: &pdf::Stream| file.decode(s).ok().and_then(|d| d.into_bytes().ok());
    match file.get(&page.dict, b"Contents") {
        Ok(Some(pdf::Object::Stream(s))) => decoded(&s).unwrap_or_default(),
        Ok(Some(pdf::Object::Array(parts))) => parts
            .iter()
            .filter_map(|p| match file.resolve(p) {
                Ok(pdf::Object::Stream(s)) => decoded(&s),
                _ => None,
            })
            .flatten()
            .collect(),
        _ => Vec::new(),
    }
}

fn main() {
    let mut args = std::env::args_os().skip(1).peekable();
    let mut flattening = Flattening::Samples(8);
    if let Some(option) = args.next_if(|a| a == "--samples" || a == "--flatness") {
        let value = args.next().and_then(|v| v.into_string().ok());
        flattening = if option == "--samples" {
            Flattening::Samples(
                value
                    .and_then(|v| v.parse().ok())
                    .expect("--samples takes a count"),
            )
        } else {
            Flattening::Flatness(
                value
                    .and_then(|v| v.parse().ok())
                    .expect("--flatness takes points"),
            )
        };
    }
    let mut files = Vec::new();
    for arg in args {
        pdfs(Path::new(&arg), &mut files);
    }
    let mut timed: Vec<(PathBuf, Stages)> = files
        .into_iter()
        .filter_map(|f| time(&f, flattening).map(|s| (f, s)))
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
