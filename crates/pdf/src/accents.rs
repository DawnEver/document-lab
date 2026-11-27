//! Accents drawn as glyphs of their own: what letter and accent read as together.
//!
//! TeX without T1 encoding draws an accented letter as two glyphs -- the letter and a spacing accent (`¨`, `´`,
//! `ˆ`, ...) set over it -- so their texts read `u` and `¨` apart. Composed, they read as the NFC form of the letter
//! and the accent's combining mark ([`compose`]): `ü`. Where an accent sits on a letter is the caller's to decide
//! ([`crate::spans`]); this module knows only the characters.

use crate::compositions::COMPOSED;

/// Each spacing accent and its combining mark: grave, acute, circumflex, tilde, macron (and the modifier letter
/// macron), breve, dot above, diaeresis, ring above, double acute, caron, cedilla, ogonek. The ASCII `^` and `~`
/// are not among them: they are characters of their own in text and code, and TeX draws its accents with the
/// glyphs `circumflex` and `tilde` (U+02C6, U+02DC).
const SPACING: [(char, char); 14] = [
    ('\u{60}', '\u{300}'),
    ('\u{B4}', '\u{301}'),
    ('\u{2C6}', '\u{302}'),
    ('\u{2DC}', '\u{303}'),
    ('\u{AF}', '\u{304}'),
    ('\u{2C9}', '\u{304}'),
    ('\u{2D8}', '\u{306}'),
    ('\u{2D9}', '\u{307}'),
    ('\u{A8}', '\u{308}'),
    ('\u{2DA}', '\u{30A}'),
    ('\u{2DD}', '\u{30B}'),
    ('\u{2C7}', '\u{30C}'),
    ('\u{B8}', '\u{327}'),
    ('\u{2DB}', '\u{328}'),
];

/// The combining mark of an accent glyph whose text is `text`: a spacing accent of [`SPACING`], its combining
/// mark itself, or the mark after a space (the accent's compatibility decomposition, as some `ToUnicode` maps
/// state it). None for any other text.
#[must_use]
pub(crate) fn mark(text: &str) -> Option<char> {
    let spacing = |c| SPACING.iter().find(|&&(s, _)| s == c).map(|&(_, m)| m);
    let combining = |c| SPACING.iter().any(|&(_, m)| m == c).then_some(c);
    let mut chars = text.chars();
    match (chars.next(), chars.next(), chars.next()) {
        (Some(c), None, _) => spacing(c).or_else(|| combining(c)),
        (Some(' '), Some(c), None) => combining(c),
        _ => None,
    }
}

/// The text of the letter `letter` with the combining `mark` over (or under) it, in NFC; None when `letter` is not
/// one letter. A dotless `ı` or `ȷ` under a mark above is the `i` or `j` TeX draws it for (`\"{\i}` is `ï`).
#[must_use]
pub(crate) fn compose(letter: &str, mark: char) -> Option<String> {
    let mut chars = letter.chars();
    let (Some(c), None) = (chars.next(), chars.next()) else {
        return None;
    };
    if !c.is_alphabetic() {
        return None;
    }
    let below = matches!(mark, '\u{327}' | '\u{328}');
    let c = match c {
        '\u{131}' if !below => 'i',
        '\u{237}' if !below => 'j',
        c => c,
    };
    Some(
        COMPOSED
            .binary_search_by(|&(l, m, _)| (l, m).cmp(&(c, mark)))
            .map_or_else(|_| format!("{c}{mark}"), |i| COMPOSED[i].2.to_string()),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_spacing_accent_its_mark_or_the_mark_after_a_space_is_an_accent() {
        assert_eq!(mark("\u{A8}"), Some('\u{308}'));
        assert_eq!(mark("\u{2C6}"), Some('\u{302}'));
        assert_eq!(mark("\u{301}"), Some('\u{301}'));
        assert_eq!(mark(" \u{308}"), Some('\u{308}'));
        assert_eq!(mark("`"), Some('\u{300}'));
        for text in ["^", "~", "a", "\u{A8}\u{A8}", " ", "", " \u{A8}", "\u{A8} "] {
            assert_eq!(mark(text), None, "{text:?}");
        }
    }

    #[test]
    fn a_letter_and_a_mark_compose_to_their_nfc_form() {
        assert_eq!(compose("u", '\u{308}').as_deref(), Some("\u{FC}"));
        assert_eq!(compose("A", '\u{30A}').as_deref(), Some("\u{C5}"));
        assert_eq!(compose("\u{FC}", '\u{301}').as_deref(), Some("\u{1D8}")); // a second accent: ǘ
        assert_eq!(compose("c", '\u{327}').as_deref(), Some("\u{E7}"));
        assert_eq!(compose("x", '\u{302}').as_deref(), Some("x\u{302}")); // no precomposed x-circumflex
        assert_eq!(compose("\u{131}", '\u{308}').as_deref(), Some("\u{EF}")); // \"{\i}
        assert_eq!(compose("\u{438}", '\u{306}').as_deref(), Some("\u{439}")); // Cyrillic short i
        assert_eq!(compose("1", '\u{308}'), None);
        assert_eq!(compose("fi", '\u{301}'), None);
        assert_eq!(compose("", '\u{301}'), None);
    }

    #[test]
    fn the_composition_table_is_sorted_for_its_search() {
        assert!(COMPOSED.is_sorted_by(|a, b| (a.0, a.1) < (b.0, b.1)));
    }
}
