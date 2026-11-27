//! A content stream's operations (ISO 32000-2 section 7.8.2): operands, then the operator they belong to.

use crate::error::{Error, Result};
use crate::object::{Dict, Object};
use crate::parser::{Parser, is_delimiter, is_whitespace};

/// One operation: an operator and its operands.
#[derive(Debug, Clone, PartialEq)]
pub struct Operation {
    /// The operator (`re`, `Tj`, `'`, ...).
    pub operator: Vec<u8>,
    /// Its operands, in order.
    pub operands: Vec<Object>,
    /// For `BI ... ID ... EI`: the inline image's dictionary and data.
    pub inline_image: Option<(Dict, Vec<u8>)>,
}

/// Every operation of `data`, and the error that stopped the reading early, if one did (the operations read
/// before it are kept).
#[must_use]
pub fn operations(data: &[u8]) -> (Vec<Operation>, Option<Error>) {
    let mut parser = Parser::new(data, 0);
    let mut out = Vec::new();
    let mut operands = Vec::new();
    loop {
        parser.skip_whitespace();
        let Some(&b) = data.get(parser.pos) else {
            return (out, None);
        };
        let operand_start = matches!(
            b,
            b'/' | b'(' | b'<' | b'[' | b'+' | b'-' | b'.' | b'0'..=b'9'
        );
        if operand_start {
            match operand(&mut parser) {
                Ok(o) => operands.push(o),
                Err(e) => return (out, Some(e)),
            }
            continue;
        }
        let at = parser.pos;
        let word = parser.token().to_vec();
        match word.as_slice() {
            b"" => {
                // A stray delimiter (`]`, `>`, `}`): skipped, as a viewer does.
                parser.pos = at + 1;
            }
            b"true" => operands.push(Object::Bool(true)),
            b"false" => operands.push(Object::Bool(false)),
            b"null" => operands.push(Object::Null),
            b"BI" => match inline_image(&mut parser) {
                Ok(image) => out.push(Operation {
                    operator: word,
                    operands: std::mem::take(&mut operands),
                    inline_image: Some(image),
                }),
                Err(e) => return (out, Some(e)),
            },
            _ => out.push(Operation {
                operator: word,
                operands: std::mem::take(&mut operands),
                inline_image: None,
            }),
        }
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
fn inline_image(parser: &mut Parser<'_>) -> Result<(Dict, Vec<u8>)> {
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
            return Ok((dict, data[begin..i - 1].to_vec()));
        }
        i += 1;
    }
    Err(Error::malformed_at(start, "an inline image without `EI`"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn operands_gather_before_their_operator() {
        let (ops, err) = operations(b"q 1 0 0 1 72 770 cm /F1 11 Tf [(a) -20 (b)] TJ Q");
        assert!(err.is_none());
        let names: Vec<_> = ops.iter().map(|o| o.operator.clone()).collect();
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
        assert_eq!(ops[1].operands.len(), 6);
        assert_eq!(ops[1].operands[0], Object::Integer(1));
    }

    #[test]
    fn integers_are_never_read_as_a_reference() {
        let (ops, _) = operations(b"0 0 R");
        assert_eq!(ops[0].operator, b"R");
        assert_eq!(ops[0].operands, [Object::Integer(0), Object::Integer(0)]);
    }

    #[test]
    fn quote_operators_are_operators() {
        let (ops, _) = operations(b"(x) ' 1 2 (y) \"");
        assert_eq!(ops[0].operator, b"'");
        assert_eq!(ops[1].operator, b"\"");
        assert_eq!(ops[1].operands.len(), 3);
    }

    #[test]
    fn an_inline_image_takes_its_data_up_to_a_lone_ei() {
        let (ops, err) = operations(b"BI /W 2 /H 1 /BPC 8 /CS /G ID \x00EI\xff\nEI Q");
        assert!(err.is_none());
        let (dict, data) = ops[0].inline_image.clone().unwrap();
        assert_eq!(dict.get(b"W"), Some(&Object::Integer(2)));
        assert_eq!(data, b"\x00EI\xff");
        assert_eq!(ops[1].operator, b"Q");
    }

    #[test]
    fn a_bad_operand_stops_with_what_was_read() {
        let (ops, err) = operations(b"q (unclosed");
        assert_eq!(ops.len(), 1);
        assert!(err.is_some());
    }
}
