//! Just enough XML (1.0, with namespaces) to read an XMP packet: elements, attributes, text, CDATA, the five
//! predefined entities and character references. A document type declaration is refused: XMP has none (XMP part 1,
//! 7.3), and refusing it keeps entity expansion out entirely.

use std::collections::HashMap;

use crate::error::{Error, Result};

/// A name resolved against its namespace: (namespace URI, local name); the URI is empty for no namespace.
pub type QName = (String, String);

/// An element.
#[derive(Debug, Clone, PartialEq)]
pub struct Element {
    /// Its name.
    pub name: QName,
    /// Its attributes, namespace declarations left out, in document order.
    pub attributes: Vec<(QName, String)>,
    /// Its content, in document order.
    pub children: Vec<Node>,
}

/// A piece of an element's content.
#[derive(Debug, Clone, PartialEq)]
pub enum Node {
    /// A child element.
    Element(Element),
    /// Character data, references resolved.
    Text(String),
}

/// The namespace `xml:` is bound to without a declaration.
const XML_NAMESPACE: &str = "http://www.w3.org/XML/1998/namespace";
/// How deep elements may nest before the document is refused as hostile.
const MAX_DEPTH: usize = 256;

impl Element {
    /// Each child element, in order.
    pub fn elements(&self) -> impl Iterator<Item = &Element> {
        self.children.iter().filter_map(|n| match n {
            Node::Element(e) => Some(e),
            Node::Text(_) => None,
        })
    }

    /// The text directly inside, concatenated.
    #[must_use]
    pub fn text(&self) -> String {
        self.children
            .iter()
            .filter_map(|n| match n {
                Node::Text(t) => Some(t.as_str()),
                Node::Element(_) => None,
            })
            .collect()
    }

    /// The attribute `namespace`:`local`, if present.
    #[must_use]
    pub fn attribute(&self, namespace: &str, local: &str) -> Option<&str> {
        self.attributes
            .iter()
            .find(|((ns, l), _)| ns == namespace && l == local)
            .map(|(_, v)| v.as_str())
    }

    /// Whether the element is `namespace`:`local`.
    #[must_use]
    pub fn is(&self, namespace: &str, local: &str) -> bool {
        self.name.0 == namespace && self.name.1 == local
    }

    /// Every element at or below this one, in document order.
    #[must_use]
    pub fn descendants(&self) -> Vec<&Element> {
        let mut out = Vec::new();
        let mut stack = vec![self];
        while let Some(e) = stack.pop() {
            out.push(e);
            let at = stack.len();
            stack.extend(e.elements());
            stack[at..].reverse();
        }
        out
    }
}

/// The root element of the XML document `text`.
///
/// # Errors
///
/// Refuses text that is not well-formed as far as this reader reads it, and a document type declaration.
pub fn parse(text: &str) -> Result<Element> {
    let mut reader = Reader { text, at: 0 };
    reader.misc()?;
    if !reader.rest().starts_with('<') {
        return Err(reader.error("no root element"));
    }
    reader.element(&mut Vec::new(), 0)
}

struct Reader<'a> {
    text: &'a str,
    at: usize,
}

/// One element's namespace bindings: prefix (empty: the default namespace) to URI.
type Scope = HashMap<String, String>;

/// A start tag as written: its name and attributes unresolved, its namespace declarations, and whether it closes
/// itself (`/>`).
struct StartTag {
    name: String,
    attributes: Vec<(String, String)>,
    scope: Scope,
    empty: bool,
}

impl Reader<'_> {
    fn rest(&self) -> &str {
        &self.text[self.at..]
    }

    fn error(&self, what: &str) -> Error {
        Error::malformed_at(self.at, format!("XML: {what}"))
    }

    /// Skip what may stand outside the root element: whitespace, comments, processing instructions.
    fn misc(&mut self) -> Result<()> {
        loop {
            self.skip_space();
            let rest = self.rest();
            if rest.starts_with("<?") {
                self.skip_past("?>")?;
            } else if rest.starts_with("<!--") {
                self.skip_past("-->")?;
            } else if rest.starts_with("<!DOCTYPE") {
                return Err(self.error("a document type declaration, which XMP has not"));
            } else {
                return Ok(());
            }
        }
    }

    fn skip_space(&mut self) {
        let trimmed = self
            .rest()
            .trim_start_matches([' ', '\t', '\r', '\n', '\u{feff}']);
        self.at = self.text.len() - trimmed.len();
    }

    fn skip_past(&mut self, end: &str) -> Result<()> {
        let found = self
            .rest()
            .find(end)
            .ok_or_else(|| self.error(&format!("no closing {end}")))?;
        self.at += found + end.len();
        Ok(())
    }

    fn name(&mut self) -> Result<&str> {
        let start = self.at;
        let len = self
            .rest()
            .find(|c: char| c.is_whitespace() || matches!(c, '=' | '/' | '>'))
            .unwrap_or(self.rest().len());
        if len == 0 {
            return Err(self.error("a name expected"));
        }
        self.at += len;
        Ok(&self.text[start..start + len])
    }

    fn start_tag(&mut self) -> Result<StartTag> {
        self.at += 1; // '<'
        let name = self.name()?.to_string();
        let mut attributes = Vec::new();
        let mut scope = Scope::new();
        loop {
            self.skip_space();
            let rest = self.rest();
            if rest.starts_with("/>") {
                self.at += 2;
                return Ok(StartTag {
                    name,
                    attributes,
                    scope,
                    empty: true,
                });
            }
            if rest.starts_with('>') {
                self.at += 1;
                return Ok(StartTag {
                    name,
                    attributes,
                    scope,
                    empty: false,
                });
            }
            let attribute = self.name()?.to_string();
            self.skip_space();
            if !self.rest().starts_with('=') {
                return Err(self.error("an attribute without its value"));
            }
            self.at += 1;
            self.skip_space();
            let value = self.quoted()?;
            if attribute == "xmlns" {
                scope.insert(String::new(), value);
            } else if let Some(prefix) = attribute.strip_prefix("xmlns:") {
                scope.insert(prefix.to_string(), value);
            } else {
                attributes.push((attribute, value));
            }
        }
    }

    fn element(&mut self, scopes: &mut Vec<Scope>, depth: usize) -> Result<Element> {
        if depth > MAX_DEPTH {
            return Err(self.error("elements nest too deep"));
        }
        let mut tag = self.start_tag()?;
        scopes.push(std::mem::take(&mut tag.scope));
        let element = self.element_in_scope(scopes, depth, &tag);
        scopes.pop();
        element
    }

    fn element_in_scope(
        &mut self,
        scopes: &mut Vec<Scope>,
        depth: usize,
        tag: &StartTag,
    ) -> Result<Element> {
        let raw_name = &tag.name;
        let name = resolve(scopes, raw_name, false)?;
        let attributes = tag
            .attributes
            .iter()
            .map(|(n, v)| Ok((resolve(scopes, n, true)?, v.clone())))
            .collect::<Result<_>>()?;
        let mut element = Element {
            name,
            attributes,
            children: Vec::new(),
        };
        if tag.empty {
            return Ok(element);
        }
        loop {
            let rest = self.rest();
            if rest.is_empty() {
                return Err(self.error(&format!("<{raw_name}> is never closed")));
            }
            if let Some(after) = rest.strip_prefix("</") {
                let len = after
                    .find('>')
                    .ok_or_else(|| self.error("an unclosed end tag"))?;
                if after[..len].trim() != raw_name {
                    return Err(
                        self.error(&format!("<{raw_name}> is closed by </{}>", &after[..len]))
                    );
                }
                self.at += 2 + len + 1;
                return Ok(element);
            }
            if rest.starts_with("<!--") {
                self.skip_past("-->")?;
            } else if rest.starts_with("<?") {
                self.skip_past("?>")?;
            } else if let Some(after) = rest.strip_prefix("<![CDATA[") {
                let len = after
                    .find("]]>")
                    .ok_or_else(|| self.error("an unclosed CDATA section"))?;
                element.children.push(Node::Text(after[..len].to_string()));
                self.at += "<![CDATA[".len() + len + 3;
            } else if rest.starts_with('<') {
                let child = self.element(scopes, depth + 1)?;
                element.children.push(Node::Element(child));
            } else {
                let len = rest.find('<').unwrap_or(rest.len());
                let text = unescape(&rest[..len]).map_err(|e| self.error(&e))?;
                element.children.push(Node::Text(text));
                self.at += len;
            }
        }
    }

    fn quoted(&mut self) -> Result<String> {
        let quote = self
            .rest()
            .chars()
            .next()
            .filter(|c| matches!(c, '"' | '\''))
            .ok_or_else(|| self.error("an attribute value is not quoted"))?;
        let after = &self.rest()[1..];
        let len = after
            .find(quote)
            .ok_or_else(|| self.error("an attribute value is never closed"))?;
        let value = unescape(&after[..len]).map_err(|e| self.error(&e))?;
        self.at += 1 + len + 1;
        Ok(value)
    }
}

/// The raw name `raw` (an attribute's when `attribute`) resolved against `scopes`, innermost last.
fn resolve(scopes: &[Scope], raw: &str, attribute: bool) -> Result<QName> {
    let (prefix, local) = raw.split_once(':').unwrap_or(("", raw));
    if prefix == "xml" {
        return Ok((XML_NAMESPACE.into(), local.into()));
    }
    if prefix.is_empty() && attribute {
        return Ok((String::new(), local.into())); // an unprefixed attribute is in no namespace
    }
    match scopes.iter().rev().find_map(|s| s.get(prefix)) {
        Some(uri) => Ok((uri.clone(), local.into())),
        None if prefix.is_empty() => Ok((String::new(), local.into())),
        None => Err(Error::malformed(format!(
            "XML: the prefix '{prefix}' is bound to no namespace"
        ))),
    }
}

/// `text` with its entity and character references resolved.
fn unescape(text: &str) -> std::result::Result<String, String> {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(amp) = rest.find('&') {
        out.push_str(&rest[..amp]);
        let after = &rest[amp + 1..];
        let end = after.find(';').ok_or("a reference without its ';'")?;
        let name = &after[..end];
        let c = match name {
            "lt" => '<',
            "gt" => '>',
            "amp" => '&',
            "quot" => '"',
            "apos" => '\'',
            _ => {
                let code = if let Some(hex) = name.strip_prefix("#x") {
                    u32::from_str_radix(hex, 16).ok()
                } else if let Some(dec) = name.strip_prefix('#') {
                    dec.parse().ok()
                } else {
                    return Err(format!("the entity &{name}; is not one of XML's five"));
                };
                code.and_then(char::from_u32)
                    .ok_or_else(|| format!("&{name}; names no character"))?
            }
        };
        out.push(c);
        rest = &after[end + 1..];
    }
    out.push_str(rest);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_resolve_by_their_namespace_and_text_by_its_references() {
        let root = parse(
            "<?xpacket begin='' id='W5M0'?><a:r xmlns:a='urn:a' xmlns='urn:d'><b x='1' a:y=\"2\">\
             t&amp;&#x41;&#66;<![CDATA[<c>]]></b><a:e/><!-- note --></a:r><?xpacket end='w'?>",
        )
        .unwrap();
        assert!(root.is("urn:a", "r"));
        let kids: Vec<&Element> = root.elements().collect();
        assert!(kids[0].is("urn:d", "b"));
        assert_eq!(kids[0].attribute("", "x"), Some("1"));
        assert_eq!(kids[0].attribute("urn:a", "y"), Some("2"));
        assert_eq!(kids[0].text(), "t&AB<c>");
        assert!(kids[1].is("urn:a", "e"));
        assert_eq!(root.descendants().len(), 3);
    }

    #[test]
    fn a_doctype_an_unknown_entity_and_a_mismatched_tag_are_refused() {
        assert!(parse("<!DOCTYPE x [<!ENTITY a 'b'>]><x>&a;</x>").is_err());
        assert!(parse("<x>&nbsp;</x>").is_err());
        assert!(parse("<x><y></x>").is_err());
        assert!(parse("<p:x/>").is_err());
    }
}
