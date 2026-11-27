//! The document outline (ISO 32000-2 section 12.3.3, bookmarks): a tree of titles, each leading to a destination
//! or doing an action. An item met twice (a loop) cuts the chain there, and a destination that does not resolve
//! leaves its item without one; each is a warning naming the item, so a hostile outline never hangs a reader.

use std::collections::HashSet;

use crate::destination::{Action, Destination, Targets};
use crate::file::File;
use crate::object::{Dict, Object, text_string};

/// How deep the outline may nest before the deeper items are cut.
const MAX_DEPTH: usize = 64;

/// One outline item.
#[derive(Debug, Clone, PartialEq)]
pub struct Item {
    /// Its title as stated; None when it states none.
    pub title: Option<String>,
    /// Where it leads in this document: its `/Dest`, or its `GoTo` action's.
    pub destination: Option<Destination>,
    /// What it does other than go to a place in this document (a `URI`, a `GoToR`, ...).
    pub action: Option<Action>,
    /// Its children, in order.
    pub children: Vec<Item>,
}

impl Item {
    /// The item as a warning names it: its title quoted, else its want of one.
    fn named(&self) -> String {
        self.title
            .as_ref()
            .map_or_else(|| "an untitled item".into(), |t| format!("'{t}'"))
    }
}

impl File {
    /// The outline's top-level items, in order (empty when the document has none); what is cut goes to `notes`.
    pub fn outline(&self, targets: &Targets, notes: &mut Vec<String>) -> Vec<Item> {
        let root = match self.catalog().and_then(|c| self.get(&c, b"Outlines")) {
            Ok(Some(Object::Dict(root))) => root,
            Ok(_) => return Vec::new(),
            Err(e) => {
                notes.push(format!("outline: /Outlines unreadable ({e})"));
                return Vec::new();
            }
        };
        let mut seen = HashSet::new();
        self.items(&root, targets, &mut seen, 0, notes)
    }

    /// The children of `parent`: its `/First` and each `/Next` after it.
    fn items(
        &self,
        parent: &Dict,
        targets: &Targets,
        seen: &mut HashSet<u32>,
        depth: usize,
        notes: &mut Vec<String>,
    ) -> Vec<Item> {
        let mut out = Vec::new();
        let mut next = parent.get(b"First").cloned();
        let mut previous = String::from("the start of a level");
        while let Some(link) = next.take() {
            if let Some(r) = link.as_reference()
                && !seen.insert(r.num)
            {
                notes.push(format!(
                    "outline: after {previous} the chain leads back to object {}; the loop is cut",
                    r.num
                ));
                break;
            }
            let dict = match self.resolve(&link) {
                Ok(Object::Dict(d)) => d,
                Ok(Object::Null) => break,
                Ok(other) => {
                    notes.push(format!(
                        "outline: after {previous} an item is a {}; the rest cut",
                        other.kind()
                    ));
                    break;
                }
                Err(e) => {
                    notes.push(format!(
                        "outline: after {previous} an item is unreadable ({e}); the rest cut"
                    ));
                    break;
                }
            };
            let item = self.item(&dict, targets, seen, depth, notes);
            previous = item.named();
            out.push(item);
            next = dict.get(b"Next").cloned();
        }
        out
    }

    fn item(
        &self,
        dict: &Dict,
        targets: &Targets,
        seen: &mut HashSet<u32>,
        depth: usize,
        notes: &mut Vec<String>,
    ) -> Item {
        let title = match self.get(dict, b"Title") {
            Ok(Some(Object::String(s))) => Some(text_string(&s)),
            _ => None,
        };
        let mut item = Item {
            title,
            destination: None,
            action: None,
            children: Vec::new(),
        };
        match targets.target(self, dict) {
            Ok((destination, action)) => (item.destination, item.action) = (destination, action),
            Err(e) => notes.push(format!(
                "outline item {}: its target dropped: {e}",
                item.named()
            )),
        }
        if dict.get(b"First").is_some() {
            if depth >= MAX_DEPTH {
                notes.push(format!(
                    "outline item {}: nests deeper than {MAX_DEPTH}; its children are cut",
                    item.named()
                ));
            } else {
                item.children = self.items(dict, targets, seen, depth + 1, notes);
            }
        }
        item
    }
}
