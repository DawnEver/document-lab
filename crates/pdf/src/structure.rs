//! The logical structure of a tagged PDF (ISO 32000-2 sections 14.7-14.8): the tree under the catalog's
//! `/StructTreeRoot`, read as elements in tree order -- the reading order the document states.
//!
//! Each element keeps its role as stated (`/S`) and the standard role it maps to: itself when standard (the
//! types of 14.8.4, PDF 1.7's and 2.0's, a heading `Hn` of any level), else followed through the role map
//! (`/RoleMap`, or its namespace's `/RoleMapNS`) to one; None when the chain reaches none or loops. Its title
//! (`/T`), language (`/Lang`), alternate description (`/Alt`), replacement text (`/ActualText`) and expansion
//! (`/E`) are kept as stated, and its attributes (14.7.6): its own (`/A`) first, then its classes' (`/C`, looked
//! up in `/ClassMap`), a class's never overriding one the element states.
//!
//! Its kids, in order, are elements, marked-content sequences ([`Kid::Marked`]: an `/MCID` on a page, in the page's
//! content stream or a Form `XObject`'s, `/Stm`) and objects ([`Kid::Object`]: an annotation, an `XObject`). A
//! sequence's page is its own `/Pg`, else the nearest element's above it, else the one the parent tree places it on.
//!
//! The parent tree (`/ParentTree`, 14.7.5.4), the reverse map from content to element, is read to place such a
//! sequence and to check the tree against: content it assigns to another element, or to none, is reported, the
//! tree kept as it states its kids. What cannot be read is left out by name in [`Structure::warnings`] -- an
//! element that is its own ancestor (a cycle), the kid of two parents, a reference to nothing or to what no kid can
//! be -- never silently, and never failing the file.

use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;

use crate::error::Result;
use crate::file::File;
use crate::geometry::{Point, Rect};
use crate::object::{Dict, Object, text_string};
use crate::page::Page;

/// How deep elements may nest before the rest is left out as a hostile file.
const MAX_DEPTH: usize = 256;
/// How deep a number tree or an attribute value may nest.
const MAX_NESTING: usize = 32;

/// The standard structure types (14.8.4): PDF 1.7's and PDF 2.0's; a heading `Hn` of any level is one too.
const STANDARD: &[&str] = &[
    "Document",
    "DocumentFragment",
    "Part",
    "Art",
    "Sect",
    "Div",
    "BlockQuote",
    "Caption",
    "TOC",
    "TOCI",
    "Index",
    "NonStruct",
    "Private",
    "Aside",
    "Title",
    "FENote",
    "Sub",
    "P",
    "H",
    "L",
    "LI",
    "Lbl",
    "LBody",
    "Table",
    "TR",
    "TH",
    "TD",
    "THead",
    "TBody",
    "TFoot",
    "Span",
    "Quote",
    "Note",
    "Reference",
    "BibEntry",
    "Code",
    "Link",
    "Annot",
    "Ruby",
    "RB",
    "RT",
    "RP",
    "Warichu",
    "WT",
    "WP",
    "Figure",
    "Formula",
    "Form",
    "Em",
    "Strong",
    "Artifact",
];

/// A tagged PDF's structure tree.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Structure {
    /// Every element, in tree order (an element before its kids); [`Kid::Element`] indexes them.
    pub elements: Vec<Element>,
    /// The elements directly under the root, in order.
    pub roots: Vec<usize>,
    /// What was left out or found inconsistent, in words.
    pub warnings: Vec<String>,
}

/// A structure element (14.7.2).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Element {
    /// Its role as stated (`/S`).
    pub role: String,
    /// The standard role it is, or maps to; None when it maps to none.
    pub standard: Option<String>,
    /// Its title (`/T`).
    pub title: Option<String>,
    /// Its language (`/Lang`).
    pub lang: Option<String>,
    /// Its alternate description (`/Alt`): what a figure shows, in words.
    pub alt: Option<String>,
    /// The text stated in place of what it covers (`/ActualText`).
    pub actual_text: Option<String>,
    /// The expansion of the abbreviation it is (`/E`).
    pub expansion: Option<String>,
    /// Its attributes: its own, then its classes'.
    pub attributes: Vec<Attribute>,
    /// The page it names (`/Pg`), else the nearest element's above it, 1-based; None when none names one.
    pub page: Option<usize>,
    /// Its kids, in order.
    pub kids: Vec<Kid>,
}

/// An attribute of an element (14.7.6): its owner (`Layout`, `Table`, `List`, ...), name and value.
#[derive(Debug, Clone, PartialEq)]
pub struct Attribute {
    /// The owner (`/O`).
    pub owner: String,
    /// The attribute's name.
    pub name: String,
    /// Its value.
    pub value: Value,
}

/// An attribute's value.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// A boolean.
    Bool(bool),
    /// An integer.
    Integer(i64),
    /// A real number.
    Real(f64),
    /// A name, without its `/`.
    Name(String),
    /// A text string.
    Text(String),
    /// An array.
    Array(Vec<Value>),
    /// A dictionary (a user property), its entries in file order.
    Dict(Vec<(String, Value)>),
}

/// A kid of an element.
#[derive(Debug, Clone, PartialEq)]
pub enum Kid {
    /// An element, by its index in [`Structure::elements`].
    Element(usize),
    /// A marked-content sequence: its page (1-based), its `/MCID`, and the Form `XObject` whose content stream holds
    /// it (None: the page's).
    Marked {
        /// The page it is drawn on.
        page: usize,
        /// Its marked-content identifier.
        mcid: i64,
        /// The Form `XObject` holding it, by object number.
        stream: Option<u32>,
    },
    /// An object: an annotation or an `XObject`, by object number; its page when named, and an annotation's
    /// `/Rect` in page space.
    Object {
        /// The page it is on, 1-based.
        page: Option<usize>,
        /// Its object number.
        object: u32,
        /// An annotation's rectangle, in page space.
        rect: Option<Rect>,
    },
}

impl File {
    /// The structure tree of the file whose pages are `pages`; None when it has none (it is not tagged).
    ///
    /// # Errors
    ///
    /// Refuses a file without a catalog; what the tree itself does wrong is a warning.
    pub fn structure(&self, pages: &[Page]) -> Result<Option<Structure>> {
        let catalog = self.catalog()?;
        let Some(root) = catalog.get(b"StructTreeRoot") else {
            return Ok(None);
        };
        let mut reader = Reader::new(self, pages);
        match self.resolve(root) {
            Ok(Object::Null) => return Ok(None),
            Ok(Object::Dict(root)) => reader.read(&root),
            Ok(other) => reader.warn(format!(
                "/StructTreeRoot is a {}, not a dictionary; no element read",
                other.kind()
            )),
            Err(e) => reader.warn(format!("/StructTreeRoot unreadable: {e}")),
        }
        Ok(Some(reader.out))
    }
}

/// Where a marked-content sequence is: its page, the Form `XObject` holding it, its `/MCID`.
type Place = (usize, Option<u32>, i64);

struct Reader<'a> {
    file: &'a File,
    pages: &'a [Page],
    /// Page numbers by their page objects' numbers.
    page_numbers: HashMap<u32, usize>,
    role_map: Dict,
    class_map: Dict,
    /// Elements by their object numbers.
    element_of: HashMap<u32, usize>,
    /// The object numbers of the elements being read: the ancestors of the next.
    ancestors: HashSet<u32>,
    /// The sequences whose page no element names, for the parent tree to place.
    pending: Vec<Pending>,
    out: Structure,
}

/// An element's marked-content sequences whose page no element names: where each goes among its kids, its
/// `/MCID` and stream.
struct Pending {
    element: usize,
    described: String,
    kids: Vec<(usize, i64, Option<u32>)>,
}

impl<'a> Reader<'a> {
    fn new(file: &'a File, pages: &'a [Page]) -> Self {
        Self {
            file,
            pages,
            page_numbers: pages
                .iter()
                .filter_map(|p| Some((p.object?, p.number)))
                .collect(),
            role_map: Dict::new(),
            class_map: Dict::new(),
            element_of: HashMap::new(),
            ancestors: HashSet::new(),
            pending: Vec::new(),
            out: Structure::default(),
        }
    }

    fn warn(&mut self, warning: String) {
        self.out.warnings.push(warning);
    }

    fn dict(&self, dict: &Dict, key: &[u8]) -> Dict {
        match self.file.get(dict, key) {
            Ok(Some(Object::Dict(d))) => d,
            _ => Dict::new(),
        }
    }

    fn read(&mut self, root: &Dict) {
        self.role_map = self.dict(root, b"RoleMap");
        self.class_map = self.dict(root, b"ClassMap");
        for kid in self.list(root.get(b"K")) {
            match self.kind(&kid, "the structure tree root") {
                Some(Item::Element(dict)) => {
                    if let Some(i) = self.element(&kid, &dict, None, 0) {
                        self.out.roots.push(i);
                    }
                }
                Some(_) => self.warn(
                    "the structure tree root holds content directly, not an element; left out"
                        .into(),
                ),
                None => {}
            }
        }
        self.parent_tree(root);
    }

    /// `value` as a list: an array's items, a lone value as one, nothing for none.
    fn list(&self, value: Option<&Object>) -> Vec<Object> {
        match value.map(|v| self.file.resolve(v)) {
            Some(Ok(Object::Array(items))) => items,
            Some(Ok(Object::Null)) | None => Vec::new(),
            Some(_) => vec![value.cloned().unwrap_or(Object::Null)],
        }
    }

    /// What the kid `value` of `parent` is; None, with a warning, when it is nothing a kid can be.
    fn kind(&mut self, value: &Object, parent: &str) -> Option<Item> {
        let named = value
            .as_reference()
            .map_or_else(String::new, |r| format!(" {} {} R", r.num, r.generation));
        match self.file.resolve(value) {
            Ok(Object::Integer(mcid)) => Some(Item::Mcid(mcid)),
            Ok(Object::Dict(dict)) => {
                let kind = dict.get(b"Type").and_then(Object::as_name);
                if kind == Some(b"MCR") || (dict.get(b"S").is_none() && dict.get(b"MCID").is_some())
                {
                    Some(Item::Reference(dict))
                } else if kind == Some(b"OBJR")
                    || (dict.get(b"S").is_none() && dict.get(b"Obj").is_some())
                {
                    Some(Item::Object(dict))
                } else {
                    Some(Item::Element(dict))
                }
            }
            Ok(Object::Null) => {
                self.warn(format!(
                    "a kid{named} of {parent} is missing (a free object or null); left out"
                ));
                None
            }
            Ok(other) => {
                self.warn(format!(
                    "a kid{named} of {parent} is a {}, which no kid can be; left out",
                    other.kind()
                ));
                None
            }
            Err(e) => {
                self.warn(format!(
                    "a kid{named} of {parent} is unreadable: {e}; left out"
                ));
                None
            }
        }
    }

    /// The page `dict`'s `/Pg` names, if it names one.
    fn page_of(&self, dict: &Dict) -> Option<usize> {
        dict.get(b"Pg")
            .and_then(Object::as_reference)
            .and_then(|r| self.page_numbers.get(&r.num).copied())
    }

    /// The element `dict` (written as `value`) and its subtree, read under a page `above`; its index.
    fn element(
        &mut self,
        value: &Object,
        dict: &Dict,
        above: Option<usize>,
        depth: usize,
    ) -> Option<usize> {
        let number = value.as_reference().map(|r| r.num);
        if let Some(n) = number {
            if self.ancestors.contains(&n) {
                self.warn(format!(
                    "structure element {n} 0 R is its own ancestor (a cycle); its repeat is left out"
                ));
                return None;
            }
            if self.element_of.contains_key(&n) {
                self.warn(format!(
                    "structure element {n} 0 R is the kid of two parents; its second place is left out"
                ));
                return None;
            }
        }
        if depth > MAX_DEPTH {
            self.warn(format!(
                "structure elements nest deeper than {MAX_DEPTH}; the rest is left out"
            ));
            return None;
        }
        let text = |key: &[u8]| match self.file.get(dict, key) {
            Ok(Some(Object::String(s))) => Some(text_string(&s)),
            _ => None,
        };
        let role = dict.get(b"S").and_then(Object::as_name).map(<[u8]>::to_vec);
        let page = self.page_of(dict).or(above);
        let index = self.out.elements.len();
        let name = number.map_or_else(String::new, |n| format!(" ({n} 0 R)"));
        let shown = String::from_utf8_lossy(role.as_deref().unwrap_or(b"")).into_owned();
        let described = format!("{shown}{name}");
        self.out.elements.push(Element {
            role: shown,
            standard: role.as_deref().and_then(|r| self.standard(r, dict)),
            title: text(b"T"),
            lang: text(b"Lang"),
            alt: text(b"Alt"),
            actual_text: text(b"ActualText"),
            expansion: text(b"E"),
            attributes: self.attributes(dict),
            page,
            kids: Vec::new(),
        });
        if role.is_none() {
            self.warn(format!(
                "structure element{name} states no role (/S); read with an empty one"
            ));
        }
        if let Some(n) = number {
            self.element_of.insert(n, index);
            self.ancestors.insert(n);
        }
        let mut kids = Vec::new();
        let mut pageless = Vec::new();
        for kid in self.list(dict.get(b"K")) {
            let Some(item) = self.kind(&kid, &described) else {
                continue;
            };
            match self.kid(&kid, item, &described, page, depth) {
                Some(Read::Kid(k)) => kids.push(k),
                Some(Read::Pageless(mcid, stream)) => pageless.push((kids.len(), mcid, stream)),
                None => {}
            }
        }
        if let Some(n) = number {
            self.ancestors.remove(&n);
        }
        self.out.elements[index].kids = kids;
        if !pageless.is_empty() {
            self.pending.push(Pending {
                element: index,
                described,
                kids: pageless,
            });
        }
        Some(index)
    }

    /// The kid `kid` (an `item`) of the element `described`, under a page `page`; None when it is left out.
    fn kid(
        &mut self,
        kid: &Object,
        item: Item,
        described: &str,
        page: Option<usize>,
        depth: usize,
    ) -> Option<Read> {
        let marked = |page: Option<usize>, mcid, stream| {
            Some(page.map_or(Read::Pageless(mcid, stream), |page| {
                Read::Kid(Kid::Marked { page, mcid, stream })
            }))
        };
        match item {
            Item::Element(d) => self
                .element(kid, &d, page, depth + 1)
                .map(|i| Read::Kid(Kid::Element(i))),
            Item::Mcid(mcid) => marked(page, mcid, None),
            Item::Reference(d) => {
                let mcid = self.file.get(&d, b"MCID").ok().flatten();
                let Some(mcid) = mcid.and_then(|m| m.as_integer()) else {
                    let named = kid
                        .as_reference()
                        .map_or_else(String::new, |r| format!(" {} {} R", r.num, r.generation));
                    self.warn(format!(
                        "the marked-content reference{named} under {described} names no /MCID; left out"
                    ));
                    return None;
                };
                let stream = d.get(b"Stm").and_then(Object::as_reference).map(|r| r.num);
                marked(self.page_of(&d).or(page), mcid, stream)
            }
            Item::Object(d) => {
                let Some(object) = d.get(b"Obj").and_then(Object::as_reference) else {
                    self.warn(format!(
                        "an object reference (/OBJR) under {described} names no /Obj; left out"
                    ));
                    return None;
                };
                let page = self.page_of(&d).or(page);
                Some(Read::Kid(Kid::Object {
                    page,
                    object: object.num,
                    rect: page.and_then(|p| self.rect(p, object.num)),
                }))
            }
        }
    }

    /// The standard role `role` is, or maps to through `element`'s namespace's role map, then the tree's.
    fn standard(&self, role: &[u8], element: &Dict) -> Option<String> {
        let namespace = match self.file.get(element, b"NS") {
            Ok(Some(Object::Dict(ns))) => self.dict(&ns, b"RoleMapNS"),
            _ => Dict::new(),
        };
        let mut current = role.to_vec();
        let mut seen = HashSet::new();
        loop {
            let name = String::from_utf8_lossy(&current).into_owned();
            if is_standard(&name) {
                return Some(name);
            }
            if !seen.insert(current.clone()) {
                return None;
            }
            let mapped = namespace
                .get(&current)
                .or_else(|| self.role_map.get(&current))
                .and_then(|m| self.file.resolve(m).ok())?;
            current = match mapped {
                Object::Name(n) => n,
                Object::Array(a) => a.first().and_then(Object::as_name)?.to_vec(),
                _ => return None,
            };
        }
    }

    /// `element`'s attributes: its own (`/A`), then those of its classes (`/C`) it does not state itself.
    fn attributes(&self, element: &Dict) -> Vec<Attribute> {
        let mut out = Vec::new();
        let mut objects: Vec<Object> = self.list(element.get(b"A"));
        for class in self.list(element.get(b"C")) {
            if let Some(name) = class.as_name()
                && let Some(found) = self.class_map.get(name)
            {
                objects.extend(self.list(Some(found)));
            }
        }
        for object in objects {
            let Ok(resolved) = self.file.resolve(&object) else {
                continue;
            };
            let Some(dict) = resolved.as_dict() else {
                continue; // a revision number
            };
            let owner = dict
                .get(b"O")
                .and_then(Object::as_name)
                .map_or_else(String::new, |o| String::from_utf8_lossy(o).into_owned());
            for (key, value) in dict.iter() {
                let name = String::from_utf8_lossy(key).into_owned();
                if key == b"O"
                    || key == b"NS"
                    || out
                        .iter()
                        .any(|a: &Attribute| a.owner == owner && a.name == name)
                {
                    continue;
                }
                if let Some(value) = self.value(value, 0) {
                    out.push(Attribute {
                        owner: owner.clone(),
                        name,
                        value,
                    });
                }
            }
        }
        out
    }

    /// An attribute's value; None for what no value is (a stream, null) or nests too deep.
    fn value(&self, value: &Object, depth: usize) -> Option<Value> {
        if depth > MAX_NESTING {
            return None;
        }
        Some(match self.file.resolve(value).ok()? {
            Object::Bool(b) => Value::Bool(b),
            Object::Integer(i) => Value::Integer(i),
            Object::Real(r) => Value::Real(r),
            Object::Name(n) => Value::Name(String::from_utf8_lossy(&n).into_owned()),
            Object::String(s) => Value::Text(text_string(&s)),
            Object::Array(items) => Value::Array(
                items
                    .iter()
                    .filter_map(|i| self.value(i, depth + 1))
                    .collect(),
            ),
            Object::Dict(d) => Value::Dict(
                d.iter()
                    .filter_map(|(k, v)| {
                        Some((
                            String::from_utf8_lossy(k).into_owned(),
                            self.value(v, depth + 1)?,
                        ))
                    })
                    .collect(),
            ),
            Object::Null | Object::Stream(_) | Object::Reference(_) => return None,
        })
    }

    /// The parent tree read (14.7.5.4): each pending sequence placed on the page whose entry names its element,
    /// then the tree checked against it -- a sequence it assigns to another element, or to none, reported.
    fn parent_tree(&mut self, root: &Dict) {
        let tree = match self.file.get(root, b"ParentTree") {
            Ok(Some(Object::Dict(tree))) => Some(tree),
            _ => None,
        };
        let mut entries = HashMap::new();
        if let Some(tree) = &tree {
            self.numbers(tree, &mut entries, &mut HashSet::new(), 0);
        }
        let mut parents: HashMap<Place, usize> = HashMap::new();
        for page in self.pages {
            if let Ok(Some(key)) = self
                .file
                .get(&page.dict, b"StructParents")
                .map(|k| k.and_then(|k| k.as_integer()))
            {
                self.parents_in(entries.get(&key), page.number, None, &mut parents);
            }
        }
        let mut streams: Vec<(usize, u32)> = self
            .out
            .elements
            .iter()
            .flat_map(|e| &e.kids)
            .filter_map(|k| match k {
                Kid::Marked {
                    page,
                    stream: Some(s),
                    ..
                } => Some((*page, *s)),
                _ => None,
            })
            .collect();
        streams.sort_unstable();
        streams.dedup();
        for (page, stream) in streams {
            if let Ok(Object::Stream(form)) = self.file.object(stream)
                && let Ok(Some(key)) = self
                    .file
                    .get(&form.dict, b"StructParents")
                    .map(|k| k.and_then(|k| k.as_integer()))
            {
                self.parents_in(entries.get(&key), page, Some(stream), &mut parents);
            }
        }
        for pending in std::mem::take(&mut self.pending) {
            let mut left = Vec::new();
            for &(at, mcid, stream) in pending.kids.iter().rev() {
                let pages: Vec<usize> = parents
                    .iter()
                    .filter(|&(&(_, s, m), &e)| s == stream && m == mcid && e == pending.element)
                    .map(|(&(page, _, _), _)| page)
                    .collect();
                if let [page] = pages[..] {
                    self.out.elements[pending.element]
                        .kids
                        .insert(at, Kid::Marked { page, mcid, stream });
                } else {
                    left.push(mcid.to_string());
                }
            }
            if !left.is_empty() {
                left.reverse();
                self.warn(format!(
                    "{}: MCID {} name no page (no /Pg on it or an element above it, none in the parent tree); left out",
                    pending.described,
                    left.join(", ")
                ));
            }
        }
        if tree.is_some() {
            self.check(&parents);
        }
    }

    /// The tree's sequences checked against the parent tree's `parents`.
    fn check(&mut self, parents: &HashMap<Place, usize>) {
        let mut other = Vec::new();
        let mut none = Vec::new();
        for (i, element) in self.out.elements.iter().enumerate() {
            for kid in &element.kids {
                let &Kid::Marked { page, mcid, stream } = kid else {
                    continue;
                };
                let mut place = format!("page {page} MCID {mcid}");
                if let Some(s) = stream {
                    let _ = write!(place, " in {s} 0 R");
                }
                match parents.get(&(page, stream, mcid)) {
                    None => none.push(place),
                    Some(&j) if j != i => other.push(format!(
                        "{place}: {}, not {}",
                        self.out.elements[j].role, element.role
                    )),
                    Some(_) => {}
                }
            }
        }
        if let Some(first) = other.first() {
            let warning = format!(
                "the parent tree assigns {} marked-content sequence(s) to another element than the tree does (first: {first}); the tree is kept",
                other.len()
            );
            self.warn(warning);
        }
        if let Some(first) = none.first() {
            let warning = format!(
                "the parent tree names no element for {} marked-content sequence(s) the tree reaches (first: {first})",
                none.len()
            );
            self.warn(warning);
        }
    }

    /// The parent tree's `entry` for the content of `page` (in `stream`): an array of elements by `/MCID`.
    fn parents_in(
        &self,
        entry: Option<&Object>,
        page: usize,
        stream: Option<u32>,
        parents: &mut HashMap<Place, usize>,
    ) {
        let Some(Ok(Object::Array(items))) = entry.map(|e| self.file.resolve(e)) else {
            return;
        };
        for (mcid, item) in items.iter().enumerate() {
            if let Some(&element) = item
                .as_reference()
                .and_then(|r| self.element_of.get(&r.num))
                && let Ok(mcid) = i64::try_from(mcid)
            {
                parents.insert((page, stream, mcid), element);
            }
        }
    }

    /// The entries of the number tree `node` (7.9.7) into `out`; a kid seen before, or past [`MAX_NESTING`], is
    /// passed over.
    fn numbers(
        &self,
        node: &Dict,
        out: &mut HashMap<i64, Object>,
        seen: &mut HashSet<u32>,
        depth: usize,
    ) {
        if depth > MAX_NESTING {
            return;
        }
        if let Ok(Some(Object::Array(nums))) = self.file.get(node, b"Nums") {
            for pair in nums.chunks(2) {
                if let [key, value] = pair
                    && let Some(key) = key.as_integer()
                {
                    out.insert(key, value.clone());
                }
            }
        }
        for kid in self.list(node.get(b"Kids")) {
            if kid.as_reference().is_some_and(|r| !seen.insert(r.num)) {
                continue;
            }
            if let Ok(Object::Dict(kid)) = self.file.resolve(&kid) {
                self.numbers(&kid, out, seen, depth + 1);
            }
        }
    }

    /// The `/Rect` of the annotation `object` on page `page`, in page space; None when it is no annotation.
    fn rect(&self, page: usize, object: u32) -> Option<Rect> {
        let Object::Dict(annotation) = self.file.object(object).ok()? else {
            return None;
        };
        let Object::Array(r) = self.file.get(&annotation, b"Rect").ok()?? else {
            return None;
        };
        let n: Vec<f64> = r.iter().filter_map(Object::as_number).collect();
        let [x0, y0, x1, y1] = n[..] else {
            return None;
        };
        let to_page = self.pages.get(page.checked_sub(1)?)?.to_page_space();
        let corners =
            [(x0, y0), (x1, y0), (x1, y1), (x0, y1)].map(|(x, y)| to_page.apply(Point { x, y }));
        Rect::around(&corners)
    }
}

/// Whether `role` is a standard structure type.
fn is_standard(role: &str) -> bool {
    STANDARD.contains(&role)
        || role.strip_prefix('H').is_some_and(|level| {
            !level.is_empty()
                && level.bytes().all(|b| b.is_ascii_digit())
                && !level.starts_with('0')
        })
}

/// A kid read: a kid, or a marked-content sequence (its `/MCID` and stream) whose page nothing above it names.
enum Read {
    Kid(Kid),
    Pageless(i64, Option<u32>),
}

/// What a kid is.
enum Item {
    Element(Dict),
    Mcid(i64),
    /// A marked-content reference (`/MCR`).
    Reference(Dict),
    /// An object reference (`/OBJR`).
    Object(Dict),
}
