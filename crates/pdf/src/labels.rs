//! Page labels (ISO 32000-2 section 12.4.2): the name each page is shown by -- `iv`, `12`, `A-3` -- from the
//! catalog's `/PageLabels` number tree of ranges, each a style, a prefix and a first number.

use crate::file::File;
use crate::object::{Object, text_string};

impl File {
    /// Each of the first `count` pages' label, in order; None when the document states no labels. A page before
    /// the first range has no label (None); a range without a style labels its pages by its prefix alone (12.4.2).
    /// What cannot be read goes to `notes`.
    pub fn page_labels(
        &self,
        count: usize,
        notes: &mut Vec<String>,
    ) -> Option<Vec<Option<String>>> {
        let root = self.catalog().ok()?.get(b"PageLabels")?.clone();
        let mut ranges: Vec<(usize, Option<u8>, String, i64)> = Vec::new();
        for (start, value) in self.number_tree(&root, "page labels", notes) {
            let Some(start) = usize::try_from(start).ok() else {
                notes.push(format!(
                    "page labels: a range starts at page index {start}; skipped"
                ));
                continue;
            };
            let Ok(Object::Dict(range)) = self.resolve(&value) else {
                notes.push(format!(
                    "page labels: the range at page index {start} is no dictionary; skipped"
                ));
                continue;
            };
            let style = match self.get(&range, b"S") {
                Ok(Some(Object::Name(n))) if matches!(&n[..], b"D" | b"R" | b"r" | b"A" | b"a") => {
                    Some(n[0])
                }
                Ok(Some(Object::Name(n))) => {
                    notes.push(format!(
                        "page labels: the range at page index {start} has style /{}, which 12.4.2 has not; prefix alone",
                        String::from_utf8_lossy(&n)
                    ));
                    None
                }
                _ => None,
            };
            let prefix = match self.get(&range, b"P") {
                Ok(Some(Object::String(s))) => text_string(&s),
                _ => String::new(),
            };
            // 12.4.2 Table 161: /St defaults to 1.
            let first = match self.get(&range, b"St") {
                Ok(Some(Object::Integer(n))) if n >= 1 => n,
                Ok(Some(other)) => {
                    notes.push(format!(
                        "page labels: the range at page index {start} starts at {other:?}, not a positive integer; read as 1"
                    ));
                    1
                }
                _ => 1,
            };
            ranges.push((start, style, prefix, first));
        }
        ranges.sort_by_key(|r| r.0);
        let mut unspelled = None;
        let labels = (0..count)
            .map(|page| {
                let (start, style, prefix, first) = ranges.iter().rev().find(|r| r.0 <= page)?;
                let n = first.saturating_add(i64::try_from(page - start).unwrap_or(i64::MAX));
                let Some(style) = style else {
                    return Some(prefix.clone());
                };
                let spelled = numeral(*style, n);
                if spelled.is_none() {
                    unspelled.get_or_insert(page);
                }
                Some(format!("{prefix}{}", spelled?))
            })
            .collect();
        if let Some(page) = unspelled {
            notes.push(format!(
                "page labels: from page index {page} a range numbers past {MAX_SPELLED} in roman numerals or letters; those pages unlabeled"
            ));
        }
        Some(labels)
    }
}

/// The largest number spelled in roman numerals or letters: past it a label would be thousands of characters (a
/// hostile `/St`).
const MAX_SPELLED: i64 = 10_000;

/// `n` (at least 1) in the page label style `style`: `D` decimal, `R`/`r` upper/lower roman, `A`/`a` upper/lower
/// letters (A..Z, then AA..ZZ, AAA..: the letter repeated); None past [`MAX_SPELLED`] in roman numerals or letters.
fn numeral(style: u8, n: i64) -> Option<String> {
    if style != b'D' && n > MAX_SPELLED {
        return None;
    }
    Some(match style {
        b'R' => roman(n),
        b'r' => roman(n).to_lowercase(),
        b'A' | b'a' => {
            let base = if style == b'A' { b'A' } else { b'a' };
            let letter = char::from(base + u8::try_from((n - 1) % 26).unwrap_or(0));
            letter
                .to_string()
                .repeat(usize::try_from((n - 1) / 26 + 1).unwrap_or(1))
        }
        _ => n.to_string(),
    })
}

/// `n` in upper-case roman numerals, thousands as repeated `M`.
fn roman(mut n: i64) -> String {
    const DIGITS: [(i64, &str); 13] = [
        (1000, "M"),
        (900, "CM"),
        (500, "D"),
        (400, "CD"),
        (100, "C"),
        (90, "XC"),
        (50, "L"),
        (40, "XL"),
        (10, "X"),
        (9, "IX"),
        (5, "V"),
        (4, "IV"),
        (1, "I"),
    ];
    let mut out = String::new();
    for (value, digits) in DIGITS {
        while n >= value {
            out.push_str(digits);
            n -= value;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::numeral;

    #[test]
    fn each_style_spells_its_numbers() {
        let spelled = |s: u8| {
            [1, 4, 9, 14, 26, 27, 28, 53]
                .map(|n| numeral(s, n).unwrap())
                .join(" ")
        };
        assert_eq!(spelled(b'D'), "1 4 9 14 26 27 28 53");
        assert_eq!(spelled(b'R'), "I IV IX XIV XXVI XXVII XXVIII LIII");
        assert_eq!(spelled(b'r'), "i iv ix xiv xxvi xxvii xxviii liii");
        assert_eq!(spelled(b'A'), "A D I N Z AA BB AAA");
        assert_eq!(spelled(b'a'), "a d i n z aa bb aaa");
        assert_eq!(numeral(b'R', 1994).as_deref(), Some("MCMXCIV"));
        assert_eq!(
            numeral(b'A', i64::MAX),
            None,
            "a hostile number is not spelled out"
        );
        assert_eq!(
            numeral(b'D', i64::MAX).as_deref(),
            Some("9223372036854775807")
        );
    }
}
