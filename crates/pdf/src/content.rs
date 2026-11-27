//! A content stream's operations (ISO 32000-2 section 7.8.2): operands, then the operator they belong to.

use crate::error::{Error, Result};
use crate::object::{Dict, Object};
use crate::parser::{Parser, is_delimiter, is_whitespace};

/// One operation: an operator and its operands, borrowed from the content stream being read.
#[derive(Debug, Clone, PartialEq)]
pub struct Operation<'a> {
    /// The operator (`re`, `Tj`, `'`, ...).
    pub operator: &'a [u8],
    /// Its operands, in order.
    pub operands: &'a [Object],
    /// For `BI ... ID ... EI`: the inline image's dictionary and data.
    pub inline_image: Option<(Dict, &'a [u8])>,
}

/// Hand every operation of `data` to `each`, in order; the error that stopped the reading early, if one did (the
/// operations read before it were handed over).
///
/// Operations are read one at a time, never gathered: a content stream holds millions, and its operands live in one
/// buffer reused from one operation to the next.
pub fn operations(data: &[u8], mut each: impl FnMut(&Operation<'_>)) -> Option<Error> {
    let mut parser = Parser::new(data, 0);
    let mut operands = Vec::new();
    loop {
        parser.skip_whitespace();
        let &b = data.get(parser.pos)?;
        let operand_start = matches!(
            b,
            b'/' | b'(' | b'<' | b'[' | b'+' | b'-' | b'.' | b'0'..=b'9'
        );
        if operand_start {
            match operand(&mut parser) {
                Ok(o) => operands.push(o),
                Err(e) => return Some(e),
            }
            continue;
        }
        let at = parser.pos;
        let word = parser.token();
        match word {
            b"" => {
                // A stray delimiter (`]`, `>`, `}`): skipped, as a viewer does.
                parser.pos = at + 1;
                continue;
            }
            b"true" => {
                operands.push(Object::Bool(true));
                continue;
            }
            b"false" => {
                operands.push(Object::Bool(false));
                continue;
            }
            b"null" => {
                operands.push(Object::Null);
                continue;
            }
            b"BI" => match inline_image(&mut parser) {
                Ok(image) => each(&Operation {
                    operator: word,
                    operands: &operands,
                    inline_image: Some(image),
                }),
                Err(e) => return Some(e),
            },
            _ => each(&Operation {
                operator: word,
                operands: &operands,
                inline_image: None,
            }),
        }
        operands.clear();
    }
}

/// An operand: like any object, except that `n g R` is never a reference in a content stream.
fn operand(parser: &mut Parser<'_>) -> Result<Object> {
    let at = parser.pos;
    if matches!(
        parser.data().get(at),
        Some(b'+' | b'-' | b'.' | b'0'..=b'9')
    ) {
        let word = parser.token();
        return crate::parser::number(word)
            .ok_or_else(|| Error::malformed_at(at, "not a number in a content stream"));
    }
    parser.object()
}

/// `BI` read: the key-value pairs up to `ID`, then the data up to an `EI` standing alone.
fn inline_image<'a>(parser: &mut Parser<'a>) -> Result<(Dict, &'a [u8])> {
    let start = parser.pos;
    let mut dict = Dict::new();
    loop {
        parser.skip_whitespace();
        let at = parser.pos;
        if parser.data().get(at) == Some(&b'/') {
            let Object::Name(key) = parser.object()? else {
                unreachable!("a `/` starts a name")
            };
            let value = operand(parser)?;
            dict.insert(key, value);
            continue;
        }
        if parser.token() == b"ID" {
            break;
        }
        return Err(Error::malformed_at(
            at,
            "an inline image's dictionary does not end in `ID`",
        ));
    }
    // One white-space byte separates `ID` from the data.
    let data = parser.data();
    let begin = parser.pos + 1;
    let mut i = begin;
    while i + 2 <= data.len() {
        if &data[i..i + 2] == b"EI"
            && i > begin
            && is_whitespace(data[i - 1])
            && data
                .get(i + 2)
                .is_none_or(|&c| is_whitespace(c) || is_delimiter(c))
        {
            parser.pos = i + 2;
            return Ok((dict, &data[begin..i - 1]));
        }
        i += 1;
    }
    Err(Error::malformed_at(start, "an inline image without `EI`"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every operation of `data` as owned values, and whether the reading stopped early.
    #[allow(clippy::type_complexity)]
    fn read(data: &[u8]) -> (Vec<(Vec<u8>, Vec<Object>, Option<(Dict, Vec<u8>)>)>, bool) {
        let mut ops = Vec::new();
        let err = operations(data, |op| {
            ops.push((
                op.operator.to_vec(),
                op.operands.to_vec(),
                op.inline_image
                    .as_ref()
                    .map(|(d, b)| (d.clone(), b.to_vec())),
            ));
        });
        (ops, err.is_some())
    }

    #[test]
    fn operands_gather_before_their_operator() {
        let (ops, stopped) = read(b"q 1 0 0 1 72 770 cm /F1 11 Tf [(a) -20 (b)] TJ Q");
        assert!(!stopped);
        let names: Vec<_> = ops.iter().map(|o| o.0.clone()).collect();
        assert_eq!(
            names,
            [
                b"q".to_vec(),
                b"cm".to_vec(),
                b"Tf".to_vec(),
                b"TJ".to_vec(),
                b"Q".to_vec()
            ]
        );
        assert_eq!(ops[1].1.len(), 6);
        assert_eq!(ops[1].1[0], Object::Integer(1));
        assert!(ops[4].1.is_empty(), "each operation has its own operands");
    }

    #[test]
    fn integers_are_never_read_as_a_reference() {
        let (ops, _) = read(b"0 0 R");
        assert_eq!(ops[0].0, b"R");
        assert_eq!(ops[0].1, [Object::Integer(0), Object::Integer(0)]);
    }

    #[test]
    fn quote_operators_are_operators() {
        let (ops, _) = read(b"(x) ' 1 2 (y) \"");
        assert_eq!(ops[0].0, b"'");
        assert_eq!(ops[1].0, b"\"");
        assert_eq!(ops[1].1.len(), 3);
    }

    #[test]
    fn an_inline_image_takes_its_data_up_to_a_lone_ei() {
        let (ops, stopped) = read(b"BI /W 2 /H 1 /BPC 8 /CS /G ID \x00EI\xff\nEI Q");
        assert!(!stopped);
        let (dict, data) = ops[0].2.clone().unwrap();
        assert_eq!(dict.get(b"W"), Some(&Object::Integer(2)));
        assert_eq!(data, b"\x00EI\xff");
        assert_eq!(ops[1].0, b"Q");
    }

    #[test]
    fn a_bad_operand_stops_with_what_was_read() {
        let (ops, stopped) = read(b"q (unclosed");
        assert_eq!(ops.len(), 1);
        assert!(stopped);
    }
}
