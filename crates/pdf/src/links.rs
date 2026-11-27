//! A page's links (ISO 32000-2 section 12.5.6.5): each `Link` annotation's rectangle as displayed and where it
//! leads. Links are data, not ink: they are read whether or not annotations' appearances are drawn.

use crate::destination::{Action, Destination, Targets};
use crate::file::File;
use crate::geometry::Rect;
use crate::object::Object;
use crate::page::Page;

/// One link.
#[derive(Debug, Clone, PartialEq)]
pub struct Link {
    /// The area that activates it, in page space.
    pub rect: Rect,
    /// Where it leads in this document: its `/Dest`, or its `GoTo` action's.
    pub destination: Option<Destination>,
    /// What it does other than go to a place in this document (a `URI`, a `GoToR`, ...).
    pub action: Option<Action>,
}

impl File {
    /// Every link on `page`, in `/Annots` order; one that cannot be read, or whose target does not resolve, is a
    /// warning in `notes` (the link kept without its target when its area reads).
    pub fn links(&self, page: &Page, targets: &Targets, notes: &mut Vec<String>) -> Vec<Link> {
        let annotations = match self.get(&page.dict, b"Annots") {
            Ok(Some(Object::Array(a))) => a,
            Ok(_) => return Vec::new(),
            Err(e) => {
                notes.push(format!("links: /Annots unreadable ({e})"));
                return Vec::new();
            }
        };
        let mut out = Vec::new();
        for (i, value) in annotations.iter().enumerate() {
            let Ok(Object::Dict(annotation)) = self.resolve(value) else {
                continue; // what is no annotation is no link either; drawing them warns of it
            };
            if annotation.get(b"Subtype").and_then(Object::as_name) != Some(b"Link") {
                continue;
            }
            let Ok(Some(rect)) = self.rect(&annotation, b"Rect") else {
                notes.push(format!("link {i}: its /Rect is not four numbers; skipped"));
                continue;
            };
            let rect = page.displayed(rect);
            let (destination, action) = targets.target(self, &annotation).unwrap_or_else(|e| {
                notes.push(format!("link {i}: its target dropped: {e}"));
                (None, None)
            });
            let link = Link {
                rect,
                destination,
                action,
            };
            out.push(link);
        }
        out
    }
}
