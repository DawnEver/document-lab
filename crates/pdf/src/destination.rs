//! Destinations and actions (ISO 32000-2 sections 12.3.2, 12.6): where an outline item or a link leads, the place
//! as a viewer shows it -- a page and a position in page space (origin top-left as displayed, y down).
//!
//! [`Targets`] holds what resolving needs, read once a document: each page's object and its matrix to page space,
//! and the named destinations (the catalog's `/Dests` and its `/Names /Dests` tree).

use std::collections::HashMap;

use crate::error::{Error, Result};
use crate::file::File;
use crate::geometry::{Matrix, Point, Rect};
use crate::object::{Dict, Object, text_string};
use crate::page::Page;

/// How many named destinations one lookup may follow (a name naming a name) before it is refused as a loop.
const MAX_NAMES: usize = 8;

/// A place in this document as a viewer shows it (12.3.2.2).
#[derive(Debug, Clone, PartialEq)]
pub struct Destination {
    /// The page, 0-based in document order. For a remote go-to the other file's page, its position not read.
    pub page: usize,
    /// How the page is fitted to the window, as stated: `XYZ`, `Fit`, `FitH`, `FitV`, `FitR`, `FitB`, `FitBH`,
    /// `FitBV`.
    pub fit: String,
    /// The window's left edge on the page, in page space; None when the destination leaves it to the viewer.
    pub x: Option<f64>,
    /// The window's top edge on the page, in page space; None when the destination leaves it to the viewer.
    pub y: Option<f64>,
    /// The zoom (`XYZ`); None when left to the viewer (stated null or 0, which 12.3.2.2 makes the same).
    pub zoom: Option<f64>,
    /// The box fitted to the window (`FitR`), in page space.
    pub rect: Option<Rect>,
}

/// An action (12.6.4), what of it says where it leads; any other kind is its type alone.
#[derive(Debug, Clone, PartialEq)]
pub struct Action {
    /// The action type as stated (`/S`): `GoTo`, `GoToR`, `URI`, `Named`, `Launch`, `JavaScript`, ...
    pub kind: String,
    /// Where a `GoTo` leads; for a `GoToR`, the other file's page when it names one by number.
    pub destination: Option<Destination>,
    /// A `URI` action's address, as stated.
    pub uri: Option<String>,
    /// The file a `GoToR`, `GoToE` or `Launch` action names.
    pub file: Option<String>,
    /// A `Named` action's name (`NextPage`, ...), or the named destination a `GoToR` leads to in the other file.
    pub name: Option<String>,
}

/// What resolving destinations needs, read once a document.
pub struct Targets {
    /// Each page's 0-based index by its object number.
    pages: HashMap<u32, usize>,
    /// Each page's matrix from user space to page space.
    frames: Vec<Matrix>,
    /// The named destinations: name to destination value, the catalog's `/Dests` before its name tree.
    named: HashMap<Vec<u8>, Object>,
}

impl Targets {
    /// What resolving needs for `pages` of `file`; what of the named destinations is cut goes to `notes`.
    #[must_use]
    pub fn new(file: &File, pages: &[Page], notes: &mut Vec<String>) -> Self {
        let mut named = HashMap::new();
        let catalog = file.catalog().ok();
        if let Some(Ok(Some(Object::Dict(dests)))) = catalog.as_ref().map(|c| file.get(c, b"Dests"))
        {
            for (k, v) in dests.iter() {
                named.insert(k.to_vec(), v.clone());
            }
        }
        if let Some(Ok(Some(Object::Dict(names)))) = catalog.as_ref().map(|c| file.get(c, b"Names"))
            && let Some(root) = names.get(b"Dests")
        {
            for (k, v) in file.name_tree(root, "the named destinations", notes) {
                named.entry(k).or_insert(v);
            }
        }
        Self {
            pages: pages
                .iter()
                .enumerate()
                .filter_map(|(i, p)| Some((p.reference?.num, i)))
                .collect(),
            frames: pages.iter().map(Page::to_page_space).collect(),
            named,
        }
    }

    /// The destination `value` states: an explicit array, a name or string naming one, or a dictionary holding it
    /// under `/D`.
    ///
    /// # Errors
    ///
    /// Refuses a destination naming no page of this document, a name no destination has, and a value that is no
    /// destination; the [`Error`] says which.
    pub fn destination(&self, file: &File, value: &Object) -> Result<Destination> {
        let mut current = file.resolve(value)?;
        for _ in 0..MAX_NAMES {
            current = match current {
                Object::Array(items) => return self.explicit(file, &items),
                Object::Dict(d) => file
                    .get(&d, b"D")?
                    .ok_or_else(|| Error::malformed("a destination dictionary without /D"))?,
                Object::Name(n) | Object::String(n) => {
                    let found = self.named.get(&n).ok_or_else(|| {
                        Error::malformed(format!("no destination is named '{}'", text_string(&n)))
                    })?;
                    file.resolve(found)?
                }
                other => {
                    return Err(Error::malformed(format!(
                        "a destination is a {}",
                        other.kind()
                    )));
                }
            };
        }
        Err(Error::malformed(
            "named destinations name each other in a loop",
        ))
    }

    /// The explicit destination `[page /Fit ...]` (Table 149), its position in page space.
    fn explicit(&self, file: &File, items: &[Object]) -> Result<Destination> {
        let page = match items.first() {
            Some(Object::Reference(r)) => *self.pages.get(&r.num).ok_or_else(|| {
                Error::malformed(format!("a destination names object {}, no page", r.num))
            })?,
            Some(Object::Integer(i)) => usize::try_from(*i)
                .ok()
                .filter(|i| *i < self.frames.len())
                .ok_or_else(|| Error::malformed(format!("a destination names page index {i}")))?,
            _ => return Err(Error::malformed("a destination names no page")),
        };
        let (fit, numbers) = fit_and_numbers(file, items)?;
        let at = |i: usize| numbers.get(i).copied().flatten();
        let frame = self.frames[page];
        let mut destination = Destination {
            page,
            fit: fit.clone(),
            x: None,
            y: None,
            zoom: None,
            rect: None,
        };
        let (left, top) = match fit.as_str() {
            "XYZ" => {
                destination.zoom = at(2).filter(|z| *z != 0.0);
                (at(0), at(1))
            }
            "FitH" | "FitBH" => (None, at(0)),
            "FitV" | "FitBV" => (at(0), None),
            "FitR" => {
                let [Some(left), Some(bottom), Some(right), Some(top)] =
                    [at(0), at(1), at(2), at(3)]
                else {
                    return Err(Error::malformed(
                        "a /FitR destination without its four numbers",
                    ));
                };
                let rect = Rect::spanning(
                    frame.apply(Point { x: left, y: bottom }),
                    frame.apply(Point { x: right, y: top }),
                );
                destination.rect = Some(rect);
                (destination.x, destination.y) = (Some(rect.x0), Some(rect.y0));
                return Ok(destination);
            }
            "Fit" | "FitB" => (None, None),
            other => {
                return Err(Error::malformed(format!(
                    "a destination fits the page by /{other}, which 12.3.2.2 has not"
                )));
            }
        };
        (destination.x, destination.y) = placed(frame, left, top);
        Ok(destination)
    }

    /// Where the outline item or annotation `dict` leads: (its `/Dest`, or its `GoTo` action's destination; its
    /// `/A` action of any other kind).
    ///
    /// # Errors
    ///
    /// Refuses a destination or action that does not resolve.
    pub fn target(
        &self,
        file: &File,
        dict: &Dict,
    ) -> Result<(Option<Destination>, Option<Action>)> {
        if let Some(dest) = dict.get(b"Dest") {
            return Ok((Some(self.destination(file, dest)?), None));
        }
        Ok(
            match dict.get(b"A").map(|a| self.action(file, a)).transpose()? {
                Some(a) if a.kind == "GoTo" => (a.destination, None),
                other => (None, other),
            },
        )
    }

    /// The action `value` states (12.6.4); a kind not read here is its type alone.
    ///
    /// # Errors
    ///
    /// Refuses a value that is no action dictionary, and a go-to whose destination does not resolve.
    pub fn action(&self, file: &File, value: &Object) -> Result<Action> {
        let Object::Dict(dict) = file.resolve(value)? else {
            return Err(Error::malformed("an action is no dictionary"));
        };
        let kind = file
            .get(&dict, b"S")?
            .and_then(|s| s.as_name().map(|n| String::from_utf8_lossy(n).into_owned()))
            .ok_or_else(|| Error::malformed("an action without its type /S"))?;
        let mut action = Action {
            kind,
            destination: None,
            uri: None,
            file: None,
            name: None,
        };
        match action.kind.as_str() {
            "GoTo" => {
                let d = dict
                    .get(b"D")
                    .ok_or_else(|| Error::malformed("a GoTo action without /D"))?;
                action.destination = Some(self.destination(file, d)?);
            }
            "GoToR" => {
                action.file = file_spec(file, &dict)?;
                match file.get(&dict, b"D")? {
                    Some(Object::Array(items)) => action.destination = Some(remote(file, &items)?),
                    Some(Object::Name(n) | Object::String(n)) => {
                        action.name = Some(text_string(&n));
                    }
                    _ => {}
                }
            }
            "GoToE" | "Launch" => action.file = file_spec(file, &dict)?,
            "URI" => {
                action.uri = match file.get(&dict, b"URI")? {
                    Some(Object::String(s)) => Some(String::from_utf8_lossy(&s).into_owned()),
                    _ => return Err(Error::malformed("a URI action without its /URI string")),
                };
            }
            "Named" => {
                action.name = file
                    .get(&dict, b"N")?
                    .and_then(|n| n.as_name().map(|n| String::from_utf8_lossy(n).into_owned()));
            }
            _ => {}
        }
        Ok(action)
    }
}

/// A destination array's fit name and its numbers after it, null ones None.
fn fit_and_numbers(file: &File, items: &[Object]) -> Result<(String, Vec<Option<f64>>)> {
    let fit = match items.get(1).map(|o| file.resolve(o)).transpose()? {
        Some(Object::Name(n)) => String::from_utf8_lossy(&n).into_owned(),
        _ => return Err(Error::malformed("a destination without its fit name")),
    };
    let numbers = items
        .iter()
        .skip(2)
        .map(|o| file.resolve(o).map(|o| o.as_number()))
        .collect::<Result<_>>()?;
    Ok((fit, numbers))
}

/// A remote go-to's explicit destination: the other file's page by number (7.11.7 makes it 0-based there), its
/// position not read.
fn remote(file: &File, items: &[Object]) -> Result<Destination> {
    let page = items
        .first()
        .and_then(Object::as_integer)
        .and_then(|i| usize::try_from(i).ok())
        .ok_or_else(|| Error::malformed("a GoToR destination names no page number"))?;
    let (fit, _) = fit_and_numbers(file, items)?;
    Ok(Destination {
        page,
        fit,
        x: None,
        y: None,
        zoom: None,
        rect: None,
    })
}

/// The file an action's `/F` names (7.11.2): a string, or a file specification's `/UF`, else its `/F`.
fn file_spec(file: &File, dict: &Dict) -> Result<Option<String>> {
    Ok(match file.get(dict, b"F")? {
        Some(Object::String(s)) => Some(text_string(&s)),
        Some(Object::Dict(spec)) => spec_name(file, &spec)?,
        _ => None,
    })
}

/// A file specification dictionary's file name: its `/UF`, else its `/F` (7.11.3).
pub(crate) fn spec_name(file: &File, spec: &Dict) -> Result<Option<String>> {
    for key in [&b"UF"[..], b"F"] {
        if let Some(Object::String(s)) = file.get(spec, key)? {
            return Ok(Some(text_string(&s)));
        }
    }
    Ok(None)
}

/// The point (`left`, `top`) of user space in page space, a coordinate left to the viewer still None: a quarter
/// turn swaps which user coordinate each page coordinate comes from.
fn placed(frame: Matrix, left: Option<f64>, top: Option<f64>) -> (Option<f64>, Option<f64>) {
    let point = frame.apply(Point {
        x: left.unwrap_or(0.0),
        y: top.unwrap_or(0.0),
    });
    let [x_from_x, _, _, y_from_y, ..] = frame.0;
    let x = if x_from_x == 0.0 { top } else { left }.map(|_| point.x);
    let y = if y_from_y == 0.0 { left } else { top }.map(|_| point.y);
    (x, y)
}
