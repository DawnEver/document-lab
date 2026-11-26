//! Optional content (ISO 32000-2 section 8.11): which content the file's default view shows.
//!
//! A group's state is set by the default configuration (`/OCProperties /D`): its `/BaseState`, then the groups its
//! `/ON` names, then those its `/OFF` names (so a group named in both is off). A group whose `/Intent` shares none
//! with the configuration's is ignored -- it shows. A membership dictionary (`/OCMD`) shows by its visibility
//! expression `/VE` when it has one, else by its policy `/P` over its `/OCGs`. A file without `/OCProperties` has no
//! optional content: everything shows.
//!
//! Not read: usage (`/Usage`) and the auto-state rules that apply it (`/AS`: zoom, print, language, ...), alternate
//! configurations (`/Configs`), and radio-button groups (`/RBGroups`, which constrain a viewer's changes, not the
//! default view).

use std::collections::HashSet;

use crate::error::{Error, Result};
use crate::file::File;
use crate::object::{Dict, Object};

/// How deep a visibility expression may nest before it is refused as a loop.
const MAX_EXPRESSION_DEPTH: usize = 16;

/// The default view of a file's optional content.
#[derive(Debug, Clone, Default)]
pub struct OptionalContent {
    /// Whether the file has optional content at all.
    present: bool,
    /// A group named in neither `/ON` nor `/OFF` is on.
    base_on: bool,
    /// The groups `/ON` names, by object number.
    on: HashSet<u32>,
    /// The groups `/OFF` names, by object number.
    off: HashSet<u32>,
    /// The intents the configuration applies (`/View` by default); None: `/All`.
    intents: Option<Vec<Vec<u8>>>,
}

impl File {
    /// The default view of this file's optional content.
    ///
    /// # Errors
    ///
    /// Refuses a catalog that cannot be read.
    pub fn optional_content(&self) -> Result<OptionalContent> {
        let catalog = self.catalog()?;
        let Some(Object::Dict(properties)) = self.get(&catalog, b"OCProperties")? else {
            return Ok(OptionalContent::default());
        };
        let config = match self.get(&properties, b"D")? {
            Some(Object::Dict(d)) => d,
            _ => Dict::new(),
        };
        let numbers = |key: &[u8]| -> Result<HashSet<u32>> {
            Ok(
                match config.get(key).map(|v| self.resolve(v)).transpose()? {
                    Some(Object::Array(groups)) => groups
                        .iter()
                        .filter_map(|g| g.as_reference().map(|r| r.num))
                        .collect(),
                    _ => HashSet::new(),
                },
            )
        };
        let base = self.get(&config, b"BaseState")?;
        Ok(OptionalContent {
            present: true,
            base_on: base.as_ref().and_then(Object::as_name) != Some(b"OFF"),
            on: numbers(b"ON")?,
            off: numbers(b"OFF")?,
            intents: intents(self.get(&config, b"Intent")?.as_ref()),
        })
    }
}

/// The intents an `/Intent` entry names (`/View` when absent, or neither a name nor an array); None for `/All`.
fn intents(value: Option<&Object>) -> Option<Vec<Vec<u8>>> {
    let names: Vec<Vec<u8>> = match value {
        Some(Object::Name(n)) => vec![n.clone()],
        Some(Object::Array(items)) => items
            .iter()
            .filter_map(|i| i.as_name().map(<[u8]>::to_vec))
            .collect(),
        _ => vec![b"View".to_vec()],
    };
    (!names.iter().any(|n| n == b"All")).then_some(names)
}

impl OptionalContent {
    /// Whether the content `oc` governs shows: `oc` is an `/OC` entry or marked-content properties -- a group or a
    /// membership dictionary, direct or by reference.
    ///
    /// # Errors
    ///
    /// Refuses `oc` when it is neither a group nor a membership dictionary, or its expression cannot be read.
    pub fn shows(&self, file: &File, oc: &Object) -> Result<bool> {
        if !self.present {
            return Ok(true);
        }
        let number = oc.as_reference().map(|r| r.num);
        let Object::Dict(dict) = file.resolve(oc)? else {
            return Err(Error::malformed(
                "optional content that is not a dictionary",
            ));
        };
        let kind = file.get(&dict, b"Type")?;
        match kind.as_ref().and_then(Object::as_name) {
            Some(b"OCG") => self.group_on(file, number, &dict),
            Some(b"OCMD") => self.member_shows(file, &dict),
            _ if dict.get(b"OCGs").is_some() || dict.get(b"VE").is_some() => {
                self.member_shows(file, &dict)
            }
            _ if dict.get(b"Name").is_some() => self.group_on(file, number, &dict),
            _ => Err(Error::malformed(
                "optional content that is neither a group nor a membership dictionary",
            )),
        }
    }

    /// Whether the group `value` (a membership's member) is on.
    fn group(&self, file: &File, value: &Object) -> Result<bool> {
        match file.resolve(value)? {
            Object::Dict(dict) => self.group_on(file, value.as_reference().map(|r| r.num), &dict),
            other => Err(Error::malformed(format!(
                "a membership names a {}, not a group",
                other.kind()
            ))),
        }
    }

    /// Whether the group `dict` (object `number`, when indirect) is on.
    fn group_on(&self, file: &File, number: Option<u32>, dict: &Dict) -> Result<bool> {
        if let Some(applied) = &self.intents {
            let own = intents(file.get(dict, b"Intent")?.as_ref()).unwrap_or_default();
            if !own.is_empty() && !own.iter().any(|i| applied.contains(i)) {
                return Ok(true); // a group of no intent the view applies is ignored
            }
        }
        Ok(match number {
            Some(n) if self.off.contains(&n) => false,
            Some(n) if self.on.contains(&n) => true,
            _ => self.base_on,
        })
    }

    /// Whether the membership dictionary `dict` shows its content.
    fn member_shows(&self, file: &File, dict: &Dict) -> Result<bool> {
        if let Some(expression) = dict.get(b"VE") {
            return self.expression(file, expression, 0);
        }
        let groups = match dict.get(b"OCGs") {
            None => Vec::new(),
            Some(value) => match file.resolve(value)? {
                Object::Array(items) => items,
                Object::Null => Vec::new(),
                _ => vec![value.clone()],
            },
        };
        let mut states = Vec::new();
        for group in &groups {
            if !matches!(file.resolve(group)?, Object::Null) {
                states.push(self.group(file, group)?);
            }
        }
        if states.is_empty() {
            return Ok(true); // a membership of no group has no effect
        }
        let policy = file.get(dict, b"P")?;
        Ok(match policy.as_ref().and_then(Object::as_name) {
            Some(b"AllOn") => states.iter().all(|&s| s),
            Some(b"AnyOff") => states.iter().any(|&s| !s),
            Some(b"AllOff") => states.iter().all(|&s| !s),
            _ => states.iter().any(|&s| s), // AnyOn, the default
        })
    }

    /// The value of the visibility expression `value` (8.11.2.2): a group, or `[/And|/Or|/Not operand ...]`.
    fn expression(&self, file: &File, value: &Object, depth: usize) -> Result<bool> {
        if depth > MAX_EXPRESSION_DEPTH {
            return Err(Error::malformed("a visibility expression nests too deep"));
        }
        let Object::Array(items) = file.resolve(value)? else {
            return self.group(file, value);
        };
        let operator = items.first().and_then(Object::as_name).unwrap_or(b"");
        let mut operands = Vec::new();
        for item in items.get(1..).unwrap_or(&[]) {
            operands.push(self.expression(file, item, depth + 1)?);
        }
        match (operator, operands.as_slice()) {
            (b"Not", [one]) => Ok(!one),
            (b"And", [_, ..]) => Ok(operands.iter().all(|&s| s)),
            (b"Or", [_, ..]) => Ok(operands.iter().any(|&s| s)),
            _ => Err(Error::malformed(format!(
                "a visibility expression /{} of {} operands",
                String::from_utf8_lossy(operator),
                operands.len()
            ))),
        }
    }
}
