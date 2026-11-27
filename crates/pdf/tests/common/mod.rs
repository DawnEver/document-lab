//! A synthetic PDF built from its objects' source text, with a correct cross-reference table.

/// A PDF whose object `i + 1` is `objects[i]` and whose catalog is object 1.
#[must_use]
pub fn pdf(objects: &[&str]) -> Vec<u8> {
    let mut out = b"%PDF-1.7\n".to_vec();
    let mut offsets = Vec::new();
    for (i, body) in objects.iter().enumerate() {
        offsets.push(out.len());
        out.extend_from_slice(format!("{} 0 obj\n{body}\nendobj\n", i + 1).as_bytes());
    }
    let at = out.len();
    out.extend_from_slice(
        format!("xref\n0 {}\n0000000000 65535 f \n", objects.len() + 1).as_bytes(),
    );
    for o in &offsets {
        out.extend_from_slice(format!("{o:010} 00000 n \n").as_bytes());
    }
    let size = objects.len() + 1;
    out.extend_from_slice(
        format!("trailer\n<< /Size {size} /Root 1 0 R >>\nstartxref\n{at}\n%%EOF\n").as_bytes(),
    );
    out
}

/// A stream object's source text holding `data`.
#[must_use]
pub fn stream(dict: &str, data: &str) -> String {
    format!(
        "<< {dict} /Length {} >>\nstream\n{data}\nendstream",
        data.len()
    )
}

/// A one-page PDF: the page has `page_extra` in its dictionary, its content is `content`, and `extra` objects
/// follow as objects 5, 6, ...
#[must_use]
pub fn one_page(page_extra: &str, content: &str, extra: &[&str]) -> Vec<u8> {
    let content = stream("", content);
    let page = format!("<< /Type /Page /Parent 2 0 R /Contents 4 0 R {page_extra} >>");
    let mut objects = vec![
        "<< /Type /Catalog /Pages 2 0 R >>".to_string(),
        "<< /Type /Pages /Kids [3 0 R] /Count 1 /MediaBox [0 0 200 100] >>".to_string(),
        page,
        content,
    ];
    objects.extend(extra.iter().map(|s| (*s).to_string()));
    pdf(&objects.iter().map(String::as_str).collect::<Vec<_>>())
}

/// A stream object's source text holding `data` in hex (`/ASCIIHexDecode`), its dictionary also stating `dict`.
#[allow(
    dead_code,
    reason = "only the test crates embedding binary data use it"
)]
#[must_use]
pub fn hex_stream(dict: &str, data: &[u8]) -> String {
    use std::fmt::Write;
    let hex = data.iter().fold(String::new(), |mut out, b| {
        let _ = write!(out, "{b:02x}");
        out
    });
    stream(
        &format!("{dict} /Filter /ASCIIHexDecode"),
        &format!("{hex}>"),
    )
}
