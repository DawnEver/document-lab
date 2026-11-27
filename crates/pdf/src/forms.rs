//! The interactive form's fields (ISO 32000-2 section 12.7): each terminal field's fully qualified name, type,
//! value and default as stated, and its widgets -- page, rectangle as displayed, appearance state, the export value
//! a button's widget stands for. Data, not appearance: nothing here is drawn.

use std::collections::{HashMap, HashSet};

use crate::file::File;
use crate::geometry::Rect;
use crate::object::{Dict, Object, text_string};
use crate::page::Page;

/// How deep the field tree may nest before the deeper fields are cut.
const MAX_DEPTH: usize = 64;

/// A field's value as stated: a text string, a name (a button's state, a choice), or several (a multiple choice).
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// A text string.
    Text(String),
    /// A name.
    Name(String),
    /// Several values.
    List(Vec<String>),
}

/// One terminal field.
#[derive(Debug, Clone, PartialEq)]
pub struct Field {
    /// Its fully qualified name (12.7.4.2): each ancestor's partial name and its own, `.`-joined; None when no
    /// node states one.
    pub name: Option<String>,
    /// Its type as stated (`/FT`, inherited): `Btn`, `Tx`, `Ch`, `Sig`; None when no node states one.
    pub kind: Option<String>,
    /// Its flags (`/Ff`, inherited); None when no node states them.
    pub flags: Option<i64>,
    /// Its value (`/V`, inherited); None when unset, and for a signature (its value is a signature dictionary).
    pub value: Option<Value>,
    /// Its default value (`/DV`, inherited), read as `value` is.
    pub default: Option<Value>,
    /// A choice field's options (`/Opt`): (export value, text shown) each, the same when one string is stated.
    pub options: Vec<(String, String)>,
    /// Its widgets, in order.
    pub widgets: Vec<Widget>,
}

/// One widget annotation of a field: where the field is shown.
#[derive(Debug, Clone, PartialEq)]
pub struct Widget {
    /// The page (0-based) whose `/Annots` hold it, else its `/P`; None when neither does.
    pub page: Option<usize>,
    /// Its rectangle, in its page's page space; None without a page or a `/Rect`.
    pub rect: Option<Rect>,
    /// Its appearance state (`/AS`): a check box's or radio button's on state or `Off`.
    pub state: Option<String>,
    /// The export value a button's widget stands for: the button field's `/Opt` entry for it, else its on
    /// appearance state (the `/AP /N` state not `Off`); None for a push button and any other field.
    pub export: Option<String>,
}

/// What a field inherits from its ancestors (12.7.4.1): type, flags, value, default, options.
#[derive(Clone, Default)]
struct Inherited {
    name: Option<String>,
    kind: Option<Object>,
    flags: Option<Object>,
    value: Option<Object>,
    default: Option<Object>,
    options: Option<Object>,
}

impl File {
    /// Every terminal field of the interactive form, in tree order (empty when the document has none); `pages`
    /// place the widgets. What cannot be read goes to `notes`.
    pub fn fields(&self, pages: &[Page], notes: &mut Vec<String>) -> Vec<Field> {
        let roots = match self
            .catalog()
            .and_then(|c| self.get(&c, b"AcroForm"))
            .and_then(|f| match f {
                Some(Object::Dict(form)) => self.get(&form, b"Fields"),
                _ => Ok(None),
            }) {
            Ok(Some(Object::Array(fields))) => fields,
            Ok(_) => return Vec::new(),
            Err(e) => {
                notes.push(format!("form: /AcroForm /Fields unreadable ({e})"));
                return Vec::new();
            }
        };
        let mut on_page = HashMap::new();
        for (i, page) in pages.iter().enumerate() {
            if let Ok(Some(Object::Array(annots))) = self.get(&page.dict, b"Annots") {
                for a in annots.iter().filter_map(Object::as_reference) {
                    on_page.entry(a.num).or_insert(i);
                }
            }
        }
        let by_reference: HashMap<u32, usize> = pages
            .iter()
            .enumerate()
            .filter_map(|(i, p)| Some((p.reference?.num, i)))
            .collect();
        let mut walk = Walk {
            file: self,
            pages,
            on_page,
            by_reference,
            seen: HashSet::new(),
            fields: Vec::new(),
            notes,
        };
        for root in &roots {
            walk.node(root, &Inherited::default(), 0);
        }
        walk.fields
    }
}

struct Walk<'a> {
    file: &'a File,
    pages: &'a [Page],
    /// The page whose `/Annots` hold each annotation, by object number.
    on_page: HashMap<u32, usize>,
    /// Each page's index by its object number.
    by_reference: HashMap<u32, usize>,
    seen: HashSet<u32>,
    fields: Vec<Field>,
    notes: &'a mut Vec<String>,
}

impl Walk<'_> {
    /// The field node `value` under what it inherits: a terminal field when no kid of it is a field (its kids,
    /// or itself, its widgets), else each kid in turn.
    fn node(&mut self, value: &Object, above: &Inherited, depth: usize) {
        let here_name = above.name.clone().unwrap_or_else(|| "a form field".into());
        if let Some(r) = value.as_reference()
            && !self.seen.insert(r.num)
        {
            self.notes.push(format!(
                "form: under {here_name} the field tree visits object {} twice; cut",
                r.num
            ));
            return;
        }
        if depth > MAX_DEPTH {
            self.notes.push(format!(
                "form: under {here_name} the field tree nests deeper than {MAX_DEPTH}; cut"
            ));
            return;
        }
        let Ok(Object::Dict(dict)) = self.file.resolve(value) else {
            self.notes.push(format!(
                "form: under {here_name} a field is no dictionary; skipped"
            ));
            return;
        };
        let mut here = above.clone();
        if let Ok(Some(Object::String(t))) = self.file.get(&dict, b"T") {
            let partial = text_string(&t);
            here.name = Some(match &above.name {
                Some(parent) => format!("{parent}.{partial}"),
                None => partial,
            });
        }
        for (key, slot) in [
            (&b"FT"[..], &mut here.kind),
            (b"Ff", &mut here.flags),
            (b"V", &mut here.value),
            (b"DV", &mut here.default),
            (b"Opt", &mut here.options),
        ] {
            if let Ok(Some(v)) = self.file.get(&dict, key) {
                *slot = Some(v);
            }
        }
        let kids = match self.file.get(&dict, b"Kids") {
            Ok(Some(Object::Array(k))) => k,
            _ => Vec::new(),
        };
        // A kid with a partial name is a field; one without is a widget of this field (12.7.4).
        let field_kids: Vec<&Object> = kids
            .iter()
            .filter(
                |k| matches!(self.file.resolve(k), Ok(Object::Dict(d)) if d.get(b"T").is_some()),
            )
            .collect();
        if !field_kids.is_empty() {
            for kid in field_kids {
                self.node(kid, &here, depth + 1);
            }
            return;
        }
        let widgets: Vec<(Option<u32>, Dict)> = if kids.is_empty() {
            vec![(value.as_reference().map(|r| r.num), dict)]
        } else {
            kids.iter()
                .filter_map(|k| match self.file.resolve(k) {
                    Ok(Object::Dict(d)) => Some((k.as_reference().map(|r| r.num), d)),
                    _ => None,
                })
                .collect()
        };
        self.terminal(&here, &widgets);
    }

    fn terminal(&mut self, here: &Inherited, widgets: &[(Option<u32>, Dict)]) {
        let name = here.name.clone();
        let kind = here
            .kind
            .as_ref()
            .and_then(Object::as_name)
            .map(|n| String::from_utf8_lossy(n).into_owned());
        let options: Vec<Object> = match &here.options {
            Some(Object::Array(o)) => o.iter().filter_map(|v| self.file.resolve(v).ok()).collect(),
            _ => Vec::new(),
        };
        let button = kind.as_deref() == Some("Btn");
        let value_of = |v: &Option<Object>| {
            if kind.as_deref() == Some("Sig") {
                None
            } else {
                v.as_ref().and_then(|v| self.value(v))
            }
        };
        let field = Field {
            name: name.clone(),
            kind: kind.clone(),
            flags: here.flags.as_ref().and_then(Object::as_integer),
            value: value_of(&here.value),
            default: value_of(&here.default),
            options: if button {
                Vec::new()
            } else {
                options.iter().filter_map(|o| self.option(o)).collect()
            },
            widgets: widgets
                .iter()
                .enumerate()
                .map(|(i, (num, dict))| {
                    let export = if button {
                        match options.get(i) {
                            Some(Object::String(s)) => Some(text_string(s)),
                            _ => self.on_state(dict),
                        }
                    } else {
                        None
                    };
                    self.widget(name.as_deref(), *num, dict, export)
                })
                .collect(),
        };
        self.fields.push(field);
    }

    fn widget(
        &mut self,
        field: Option<&str>,
        num: Option<u32>,
        dict: &Dict,
        export: Option<String>,
    ) -> Widget {
        let page = num.and_then(|n| self.on_page.get(&n).copied()).or_else(|| {
            dict.get(b"P")
                .and_then(Object::as_reference)
                .and_then(|r| self.by_reference.get(&r.num).copied())
        });
        let rect = match (page, self.file.rect(dict, b"Rect")) {
            (Some(p), Ok(Some(r))) => Some(self.pages[p].displayed(r)),
            (_, Ok(_)) => None,
            (_, Err(e)) => {
                let field = field.unwrap_or("without a name");
                self.notes.push(format!(
                    "form field '{field}': a widget's /Rect unreadable ({e})"
                ));
                None
            }
        };
        let state = match self.file.get(dict, b"AS") {
            Ok(Some(Object::Name(n))) => Some(String::from_utf8_lossy(&n).into_owned()),
            _ => None,
        };
        Widget {
            page,
            rect,
            state,
            export,
        }
    }

    /// A widget's on appearance state: the name of its `/AP /N` (else `/D`) state other than `Off`.
    fn on_state(&self, dict: &Dict) -> Option<String> {
        let Ok(Some(Object::Dict(ap))) = self.file.get(dict, b"AP") else {
            return None;
        };
        [&b"N"[..], b"D"]
            .into_iter()
            .find_map(|key| match self.file.get(&ap, key) {
                Ok(Some(Object::Dict(states))) => states
                    .iter()
                    .map(|(k, _)| k)
                    .find(|k| *k != b"Off")
                    .map(|k| String::from_utf8_lossy(k).into_owned()),
                _ => None,
            })
    }

    fn value(&self, v: &Object) -> Option<Value> {
        match self.file.resolve(v).ok()? {
            Object::String(s) => Some(Value::Text(text_string(&s))),
            Object::Name(n) => Some(Value::Name(String::from_utf8_lossy(&n).into_owned())),
            Object::Array(items) => Some(Value::List(
                items
                    .iter()
                    .filter_map(|i| match self.file.resolve(i).ok()? {
                        Object::String(s) => Some(text_string(&s)),
                        Object::Name(n) => Some(String::from_utf8_lossy(&n).into_owned()),
                        _ => None,
                    })
                    .collect(),
            )),
            Object::Stream(s) => self
                .file
                .decode(&s)
                .and_then(crate::filter::Decoded::into_bytes)
                .ok()
                .map(|b| Value::Text(text_string(&b))),
            _ => None,
        }
    }

    /// A choice option: (export value, text shown).
    fn option(&self, o: &Object) -> Option<(String, String)> {
        match o {
            Object::String(s) => Some((text_string(s), text_string(s))),
            Object::Array(pair) => match (
                pair.first().map(|v| self.file.resolve(v)),
                pair.get(1).map(|v| self.file.resolve(v)),
            ) {
                (Some(Ok(Object::String(e))), Some(Ok(Object::String(d)))) => {
                    Some((text_string(&e), text_string(&d)))
                }
                _ => None,
            },
            _ => None,
        }
    }
}
