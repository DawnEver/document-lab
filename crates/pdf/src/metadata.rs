//! The document's metadata as stated, in both places a PDF states it (ISO 32000-2 section 14.3): the trailer's
//! document information dictionary (`/Info`, 14.3.3) and the catalog's XMP stream (`/Metadata`, 14.3.2). Neither is
//! preferred: both are reported, and [`Newer`] says which states the later date when both state one.

use std::cmp::Ordering;

use crate::file::File;
use crate::object::{Object, text_string};
use crate::xml::{self, Element};

/// The XMP namespaces read here.
const RDF: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#";
const DC: &str = "http://purl.org/dc/elements/1.1/";
const XMP: &str = "http://ns.adobe.com/xap/1.0/";
const PDF: &str = "http://ns.adobe.com/pdf/1.3/";

/// Seconds two dates may be apart and still be in either order when one does not state its offset from UT:
/// offsets run from -12:00 to +14:00.
const OFFSET_SPREAD: i64 = 26 * 3600;

/// Both metadata records, and which is newer.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Metadata {
    /// The document information dictionary's entries as stated (key, value), in file order; None when the file
    /// has none.
    pub info: Option<Vec<(String, String)>>,
    /// The XMP packet; None when the catalog has none.
    pub xmp: Option<Xmp>,
    /// Which record states the later date; None when either states none, or the two cannot be ordered.
    pub newer: Option<Newer>,
}

/// What the XMP packet states, each value as stated.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Xmp {
    /// `dc:title`: its `x-default` alternative, else its first.
    pub title: Option<String>,
    /// `dc:creator`, in order.
    pub creators: Vec<String>,
    /// `dc:description`: its `x-default` alternative, else its first.
    pub description: Option<String>,
    /// `xmp:CreateDate`.
    pub created: Option<String>,
    /// `xmp:ModifyDate`.
    pub modified: Option<String>,
    /// `xmp:MetadataDate`.
    pub metadata_date: Option<String>,
    /// `xmp:CreatorTool`.
    pub creator_tool: Option<String>,
    /// `pdf:Producer`.
    pub producer: Option<String>,
    /// `pdf:Keywords`.
    pub keywords: Option<String>,
    /// The packet's text, whole: what else it states.
    pub packet: String,
}

/// Which metadata record states the later date.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Newer {
    /// The document information dictionary.
    Info,
    /// The XMP packet.
    Xmp,
    /// Both state the same latest date.
    Same,
}

impl File {
    /// The document's metadata; what cannot be read goes to `notes`.
    pub fn metadata(&self, notes: &mut Vec<String>) -> Metadata {
        let info = match self.get(self.trailer(), b"Info") {
            Ok(Some(Object::Dict(d))) => Some(
                d.iter()
                    .filter_map(|(k, v)| {
                        let key = String::from_utf8_lossy(k).into_owned();
                        match self.resolve(v) {
                            Ok(Object::String(s)) => Some((key, text_string(&s))),
                            Ok(Object::Name(n)) => {
                                Some((key, String::from_utf8_lossy(&n).into_owned()))
                            }
                            Ok(Object::Null) => None,
                            Ok(other) => {
                                notes.push(format!(
                                    "metadata: /Info /{key} is a {}; skipped",
                                    other.kind()
                                ));
                                None
                            }
                            Err(e) => {
                                notes.push(format!("metadata: /Info /{key} unreadable ({e})"));
                                None
                            }
                        }
                    })
                    .collect(),
            ),
            Ok(_) => None,
            Err(e) => {
                notes.push(format!("metadata: /Info unreadable ({e})"));
                None
            }
        };
        let xmp = self.xmp(notes);
        let info_dates = info
            .as_ref()
            .map_or_else(Vec::new, |entries: &Vec<(String, String)>| {
                entries
                    .iter()
                    .filter(|(k, _)| k == "CreationDate" || k == "ModDate")
                    .filter_map(|(k, v)| dated(pdf_date(v), &format!("/Info /{k}"), v, notes))
                    .collect()
            });
        let xmp_dates = xmp.as_ref().map_or_else(Vec::new, |x| {
            [
                ("xmp:CreateDate", &x.created),
                ("xmp:ModifyDate", &x.modified),
                ("xmp:MetadataDate", &x.metadata_date),
            ]
            .into_iter()
            .filter_map(|(k, v)| {
                let v = v.as_ref()?;
                dated(xmp_date(v), k, v, notes)
            })
            .collect()
        });
        let newer = match (latest(&info_dates), latest(&xmp_dates)) {
            (Some(i), Some(x)) => match order(i, x) {
                Some(Ordering::Greater) => Some(Newer::Info),
                Some(Ordering::Less) => Some(Newer::Xmp),
                Some(Ordering::Equal) => Some(Newer::Same),
                None => None,
            },
            _ => None,
        };
        Metadata { info, xmp, newer }
    }

    fn xmp(&self, notes: &mut Vec<String>) -> Option<Xmp> {
        let stream = match self.catalog().and_then(|c| self.get(&c, b"Metadata")) {
            Ok(Some(Object::Stream(s))) => s,
            Ok(_) => return None,
            Err(e) => {
                notes.push(format!("metadata: the XMP stream unreadable ({e})"));
                return None;
            }
        };
        let bytes = match self
            .decode(&stream)
            .and_then(crate::filter::Decoded::into_bytes)
        {
            Ok(b) => b,
            Err(e) => {
                notes.push(format!("metadata: the XMP stream does not decode ({e})"));
                return None;
            }
        };
        let packet = unicode(&bytes);
        let mut xmp = Xmp {
            packet,
            ..Xmp::default()
        };
        match xml::parse(&xmp.packet) {
            Ok(root) => read_properties(&root, &mut xmp),
            Err(e) => notes.push(format!(
                "metadata: the XMP packet is not read ({e}); its text kept"
            )),
        }
        Some(xmp)
    }
}

/// The packet's text: UTF-16 after its byte order mark, else UTF-8 (XMP part 1, 7.1).
fn unicode(bytes: &[u8]) -> String {
    let units = |big: bool| {
        bytes[2..].chunks(2).map(move |p| {
            let pair = [p[0], p.get(1).copied().unwrap_or(0)];
            if big {
                u16::from_be_bytes(pair)
            } else {
                u16::from_le_bytes(pair)
            }
        })
    };
    match bytes {
        [0xFE, 0xFF, ..] => char::decode_utf16(units(true))
            .map(|c| c.unwrap_or('\u{FFFD}'))
            .collect(),
        [0xFF, 0xFE, ..] => char::decode_utf16(units(false))
            .map(|c| c.unwrap_or('\u{FFFD}'))
            .collect(),
        _ => String::from_utf8_lossy(bytes).into_owned(),
    }
}

/// Fill `xmp` from every `rdf:Description` under `root`: a property as an attribute, or as a child element (its
/// text, or the `rdf:li` items of its `rdf:Alt`, `rdf:Seq` or `rdf:Bag`).
fn read_properties(root: &Element, xmp: &mut Xmp) {
    for description in root
        .descendants()
        .into_iter()
        .filter(|e| e.is(RDF, "Description"))
    {
        let simple = |ns: &str, local: &str| -> Option<String> {
            description
                .attribute(ns, local)
                .map(str::to_string)
                .or_else(|| {
                    description
                        .elements()
                        .find(|e| e.is(ns, local))
                        .map(|e| e.text().trim().to_string())
                })
        };
        let items = |ns: &str, local: &str| -> Option<Vec<&Element>> {
            let property = description.elements().find(|e| e.is(ns, local))?;
            Some(
                property
                    .elements()
                    .flat_map(Element::elements)
                    .filter(|e| e.is(RDF, "li"))
                    .collect(),
            )
        };
        let alternative = |ns: &str, local: &str| -> Option<String> {
            let lis = items(ns, local)?;
            let default = lis.iter().find(|li| {
                li.attribute("http://www.w3.org/XML/1998/namespace", "lang") == Some("x-default")
            });
            default
                .or(lis.first())
                .map(|li| li.text().trim().to_string())
        };
        let fill = |slot: &mut Option<String>, value: Option<String>| {
            if slot.is_none() {
                *slot = value;
            }
        };
        fill(&mut xmp.title, alternative(DC, "title"));
        fill(&mut xmp.description, alternative(DC, "description"));
        fill(&mut xmp.created, simple(XMP, "CreateDate"));
        fill(&mut xmp.modified, simple(XMP, "ModifyDate"));
        fill(&mut xmp.metadata_date, simple(XMP, "MetadataDate"));
        fill(&mut xmp.creator_tool, simple(XMP, "CreatorTool"));
        fill(&mut xmp.producer, simple(PDF, "Producer"));
        fill(&mut xmp.keywords, simple(PDF, "Keywords"));
        if xmp.creators.is_empty()
            && let Some(lis) = items(DC, "creator")
        {
            xmp.creators = lis.iter().map(|li| li.text().trim().to_string()).collect();
        }
    }
}

/// A date: its wall-clock time in seconds from 1970-01-01 as if at UT, and its offset from UT in seconds when
/// stated.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Stamp {
    wall: i64,
    offset: Option<i64>,
}

/// `stamp`, or a note that `value` (the entry `what`) is no date.
fn dated(stamp: Option<Stamp>, what: &str, value: &str, notes: &mut Vec<String>) -> Option<Stamp> {
    if stamp.is_none() {
        notes.push(format!(
            "metadata: {what} '{value}' is no date; not compared"
        ));
    }
    stamp
}

/// The order of `a` and `b` in time; None when they cannot be told apart because an offset is not stated.
fn order(a: Stamp, b: Stamp) -> Option<Ordering> {
    match (a.offset, b.offset) {
        (Some(oa), Some(ob)) => Some((a.wall - oa).cmp(&(b.wall - ob))),
        _ if (a.wall - b.wall).abs() > OFFSET_SPREAD => Some(a.wall.cmp(&b.wall)),
        _ if a == b => Some(Ordering::Equal),
        _ => None,
    }
}

/// The latest of `stamps` by wall-clock time (one record's dates share its writer's offset).
fn latest(stamps: &[Stamp]) -> Option<Stamp> {
    stamps
        .iter()
        .copied()
        .max_by_key(|s| s.wall - s.offset.unwrap_or(0))
}

/// A PDF date (7.9.4): `D:YYYYMMDDHHmmSSOHH'mm`, everything after the year optional.
fn pdf_date(text: &str) -> Option<Stamp> {
    let t = text.trim();
    let t = t.strip_prefix("D:").unwrap_or(t);
    let digits = t.find(|c: char| !c.is_ascii_digit()).unwrap_or(t.len());
    let (fields, zone) = t.split_at(digits);
    if fields.len() < 4 || fields.len() % 2 != 0 || fields.len() > 14 {
        return None;
    }
    let part = |at: usize, default: i64| {
        fields
            .get(at..at + 2)
            .map_or(Some(default), |s| s.parse().ok())
    };
    let year: i64 = fields[..4].parse().ok()?;
    let wall = civil(
        year,
        part(4, 1)?,
        part(6, 1)?,
        part(8, 0)?,
        part(10, 0)?,
        part(12, 0)?,
    )?;
    let offset = match zone.chars().next() {
        None => None,
        Some('Z') => Some(0),
        Some(sign @ ('+' | '-')) => {
            let numbers: Vec<i64> = zone[1..]
                .split(['\'', ':'])
                .filter(|s| !s.is_empty())
                .map(|s| s.parse().ok())
                .collect::<Option<_>>()?;
            let seconds = numbers.first().copied().unwrap_or(0) * 3600
                + numbers.get(1).copied().unwrap_or(0) * 60;
            Some(if sign == '-' { -seconds } else { seconds })
        }
        Some(_) => return None,
    };
    Some(Stamp { wall, offset })
}

/// An XMP date (XMP part 1, 8.2.1.1, ISO 8601): `YYYY[-MM[-DD[Thh:mm[:ss[.s]][TZD]]]]`.
fn xmp_date(text: &str) -> Option<Stamp> {
    let t = text.trim();
    let (date, time) = t.split_once('T').unwrap_or((t, ""));
    let mut ymd = date.split('-').map(str::parse::<i64>);
    let year = ymd.next()?.ok()?;
    let month = ymd.next().transpose().ok()?.unwrap_or(1);
    let day = ymd.next().transpose().ok()?.unwrap_or(1);
    let zone_at = time.find(['Z', '+', '-']).unwrap_or(time.len());
    let (clock, zone) = time.split_at(zone_at);
    let mut hms = clock.split(':').filter(|s| !s.is_empty());
    let hour = hms.next().map_or(Some(0), |s| s.parse().ok())?;
    let minute = hms.next().map_or(Some(0), |s| s.parse().ok())?;
    let second = hms.next().map_or(Some(0.0), |s| s.parse::<f64>().ok())?;
    #[expect(
        clippy::cast_possible_truncation,
        reason = "whole seconds, a fraction dropped"
    )]
    let wall = civil(year, month, day, hour, minute, second.trunc() as i64)?;
    let offset = match zone.chars().next() {
        None => None,
        Some('Z') => Some(0),
        Some(sign) => {
            let (h, m) = zone[1..].split_once(':').unwrap_or((&zone[1..], "0"));
            let seconds = h.parse::<i64>().ok()? * 3600 + m.parse::<i64>().ok()? * 60;
            Some(if sign == '-' { -seconds } else { seconds })
        }
    };
    Some(Stamp { wall, offset })
}

/// Seconds from 1970-01-01 00:00 to the given civil time, read as UT; None for a field out of its range.
fn civil(year: i64, month: i64, day: i64, hour: i64, minute: i64, second: i64) -> Option<i64> {
    if !(1..=12).contains(&month) || !(1..=31).contains(&day) || !(0..=23).contains(&hour) {
        return None;
    }
    if !(0..=59).contains(&minute) || !(0..=60).contains(&second) {
        return None;
    }
    // Days from the civil date (H. Hinnant's algorithm, proleptic Gregorian).
    let y = if month <= 2 { year - 1 } else { year };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = (month + 9) % 12;
    let doy = (153 * mp + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146_097 + doe - 719_468;
    Some(days * 86_400 + hour * 3600 + minute * 60 + second)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dates_read_in_both_formats_and_order_by_their_offsets() {
        let pdf = pdf_date("D:20240102030405+01'00'").unwrap();
        let xmp = xmp_date("2024-01-02T02:04:05Z").unwrap();
        assert_eq!(order(pdf, xmp), Some(Ordering::Equal));
        assert_eq!(
            pdf_date("D:1970").unwrap(),
            Stamp {
                wall: 0,
                offset: None
            }
        );
        assert_eq!(
            xmp_date("1970-01-01T00:00:01.5-00:30").unwrap(),
            Stamp {
                wall: 1,
                offset: Some(-1800)
            }
        );
        assert_eq!(pdf_date("D:20241"), None);
        assert_eq!(xmp_date("2024-13-01"), None);
        // Without an offset two dates within a day of each other cannot be ordered; further apart they can.
        let local = pdf_date("D:20240102030405").unwrap();
        assert_eq!(order(local, xmp), None);
        assert_eq!(
            order(pdf_date("D:20240105").unwrap(), xmp),
            Some(Ordering::Greater)
        );
    }
}
