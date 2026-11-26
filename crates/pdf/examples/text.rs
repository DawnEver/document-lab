//! Print each page's text as the reader finds it: glyphs in drawing order, a line per text event.
//!
//! `cargo run --release --example text -- <file.pdf>`; a page starts with a `\x0c` line.

fn main() {
    let path = std::env::args_os().nth(1).expect("a PDF path");
    let file = pdf::File::open(std::fs::read(path).expect("readable"), b"").expect("a PDF");
    for page in file.pages().expect("a page tree") {
        println!("\x0c{}", page.number);
        let Ok(content) = file.interpret(&page, false) else {
            continue;
        };
        for event in content.events {
            if let pdf::Event::Text(text) = event {
                let line: String = text
                    .glyphs
                    .iter()
                    .map(|g| g.text.clone().unwrap_or_else(|| "\u{FFFD}".to_string()))
                    .collect();
                println!("{line}");
            }
        }
    }
}
