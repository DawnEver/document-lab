//! The page tree (ISO 32000-2 section 7.7.3): each page in order, its inherited attributes resolved, and the
//! matrix from its user space to page space -- PDF points, origin at the top-left of the page as displayed (its
//! crop box, turned by `/Rotate`), y growing down.

use std::collections::HashSet;

use crate::error::{Error, Result};
use crate::file::File;
use crate::geometry::{Matrix, Rect};
use crate::object::{Dict, Object};

/// How deep the page tree may nest before it is refused as a loop or a hostile file.
const MAX_DEPTH: usize = 64;

/// One page and what it inherits.
#[derive(Debug, Clone)]
pub struct Page {
    /// 1-based, in document order.
    pub number: usize,
    /// The page dictionary.
    pub dict: Dict,
    /// Its resources (inherited when the page names none).
    pub resources: Dict,
    /// The media box, in user space.
    pub media_box: Rect,
    /// The crop box (the media box when absent), clipped to the media box.
    pub crop_box: Rect,
    /// Clockwise turns of 90 degrees the page is displayed with: 0-3.
    pub quarter_turns: u8,
}

impl Page {
    /// The page's size as displayed: (width, height) in points.
    #[must_use]
    pub fn size(&self) -> (f64, f64) {
        let (w, h) = (self.crop_box.width(), self.crop_box.height());
        if self.quarter_turns % 2 == 1 {
            (h, w)
        } else {
            (w, h)
        }
    }

    /// The matrix from user space to page space (y down, origin top-left as displayed).
    #[must_use]
    pub fn to_page_space(&self) -> Matrix {
        let Rect { x0, y0, x1, y1 } = self.crop_box;
        Matrix(match self.quarter_turns {
            1 => [0.0, 1.0, 1.0, 0.0, -y0, -x0],
            2 => [-1.0, 0.0, 0.0, 1.0, x1, -y0],
            3 => [0.0, -1.0, -1.0, 0.0, y1, x1],
            _ => [1.0, 0.0, 0.0, -1.0, -x0, y1],
        })
    }
}

/// What a page inherits from the nodes above it.
#[derive(Clone, Default)]
struct Inherited {
    resources: Option<Dict>,
    media_box: Option<Rect>,
    crop_box: Option<Rect>,
    rotate: Option<i64>,
}

impl File {
    /// Every page, in order.
    ///
    /// # Errors
    ///
    /// Refuses a catalog without a page tree, a tree that loops, and a page without a media box.
    pub fn pages(&self) -> Result<Vec<Page>> {
        let root = self
            .get(&self.catalog()?, b"Pages")?
            .ok_or_else(|| Error::malformed("the catalog names no /Pages"))?;
        let Object::Dict(root) = root else {
            return Err(Error::malformed(format!(
                "/Pages is a {}, not a dictionary",
                root.kind()
            )));
        };
        let mut pages = Vec::new();
        let mut seen = HashSet::new();
        self.walk(&root, &Inherited::default(), &mut pages, &mut seen, 0)?;
        Ok(pages)
    }

    fn walk(
        &self,
        node: &Dict,
        above: &Inherited,
        pages: &mut Vec<Page>,
        seen: &mut HashSet<u32>,
        depth: usize,
    ) -> Result<()> {
        if depth > MAX_DEPTH {
            return Err(Error::malformed("the page tree nests too deep"));
        }
        let mut here = above.clone();
        if let Some(Object::Dict(r)) = self.get(node, b"Resources")? {
            here.resources = Some(r);
        }
        if let Some(r) = self.rect(node, b"MediaBox")? {
            here.media_box = Some(r);
        }
        if let Some(r) = self.rect(node, b"CropBox")? {
            here.crop_box = Some(r);
        }
        if let Some(r) = self.get(node, b"Rotate")?.and_then(|o| o.as_integer()) {
            here.rotate = Some(r);
        }
        let kids = match self.get(node, b"Kids")? {
            Some(Object::Array(kids)) => kids,
            _ if node.get(b"Type").and_then(Object::as_name) == Some(b"Pages") => Vec::new(),
            _ => {
                self.leaf(node, &here, pages);
                return Ok(());
            }
        };
        for kid in &kids {
            if let Some(r) = kid.as_reference()
                && !seen.insert(r.num)
            {
                return Err(Error::malformed(format!(
                    "the page tree visits object {} twice",
                    r.num
                )));
            }
            match self.resolve(kid)? {
                Object::Dict(d) => self.walk(&d, &here, pages, seen, depth + 1)?,
                Object::Null => {}
                other => {
                    return Err(Error::malformed(format!(
                        "a page tree node is a {}",
                        other.kind()
                    )));
                }
            }
        }
        Ok(())
    }

    fn leaf(&self, dict: &Dict, inherited: &Inherited, pages: &mut Vec<Page>) {
        let number = pages.len() + 1;
        // A page without a media box is read as US Letter, as every viewer does (7.7.3.3 makes it required).
        let media_box = inherited.media_box.unwrap_or_else(|| {
            self.repaired(format!("page {number}: no /MediaBox; read as US Letter"));
            Rect {
                x0: 0.0,
                y0: 0.0,
                x1: 612.0,
                y1: 792.0,
            }
        });
        let crop_box = inherited.crop_box.map_or(media_box, |c| Rect {
            x0: c.x0.max(media_box.x0),
            y0: c.y0.max(media_box.y0),
            x1: c.x1.min(media_box.x1),
            y1: c.y1.min(media_box.y1),
        });
        let crop_box = if crop_box.width() > 0.0 && crop_box.height() > 0.0 {
            crop_box
        } else {
            media_box
        };
        let turns = inherited.rotate.unwrap_or(0).rem_euclid(360) / 90;
        pages.push(Page {
            number,
            dict: dict.clone(),
            resources: inherited.resources.clone().unwrap_or_default(),
            media_box,
            crop_box,
            quarter_turns: u8::try_from(turns).unwrap_or(0),
        });
    }

    /// `dict[key]` as a rectangle, normalized; None when absent or not four numbers.
    pub(crate) fn rect(&self, dict: &Dict, key: &[u8]) -> Result<Option<Rect>> {
        let Some(Object::Array(items)) = self.get(dict, key)? else {
            return Ok(None);
        };
        let numbers: Vec<f64> = items
            .iter()
            .filter_map(|i| self.resolve(i).ok()?.as_number())
            .collect();
        Ok(match numbers[..] {
            [a, b, c, d] => Some(Rect {
                x0: a.min(c),
                y0: b.min(d),
                x1: a.max(c),
                y1: b.max(d),
            }),
            _ => None,
        })
    }
}
