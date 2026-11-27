//! Name trees and number trees (ISO 32000-2 sections 7.9.6, 7.9.7): sorted keys to values, spread over nodes. A
//! node met twice, a tree nested past `MAX_DEPTH` and a node that is no dictionary are cut, each in a warning.

use std::collections::HashSet;

use crate::file::File;
use crate::object::Object;

/// How deep a tree may nest before it is cut as a loop or a hostile file.
const MAX_DEPTH: usize = 64;

impl File {
    /// Every key and value of the name tree `root`, in the order the tree stores them; values unresolved. What is
    /// cut goes to `notes`, naming the tree by `what`.
    pub fn name_tree(
        &self,
        root: &Object,
        what: &str,
        notes: &mut Vec<String>,
    ) -> Vec<(Vec<u8>, Object)> {
        self.tree(root, b"Names", what, notes, |key| match key {
            Object::String(s) => Some(s.clone()),
            _ => None,
        })
    }

    /// Every key and value of the number tree `root`, in the order the tree stores them; values unresolved. What
    /// is cut goes to `notes`, naming the tree by `what`.
    pub fn number_tree(
        &self,
        root: &Object,
        what: &str,
        notes: &mut Vec<String>,
    ) -> Vec<(i64, Object)> {
        self.tree(root, b"Nums", what, notes, Object::as_integer)
    }

    fn tree<K>(
        &self,
        root: &Object,
        leaves: &[u8],
        what: &str,
        notes: &mut Vec<String>,
        key: impl Fn(&Object) -> Option<K>,
    ) -> Vec<(K, Object)> {
        let mut out = Vec::new();
        let mut seen = HashSet::new();
        let mut stack = vec![(root.clone(), 0)];
        while let Some((node, depth)) = stack.pop() {
            if let Some(r) = node.as_reference()
                && !seen.insert(r.num)
            {
                notes.push(format!(
                    "{what}: the tree visits object {} twice; the loop is cut",
                    r.num
                ));
                continue;
            }
            if depth > MAX_DEPTH {
                notes.push(format!(
                    "{what}: the tree nests deeper than {MAX_DEPTH}; cut"
                ));
                continue;
            }
            let dict = match self.resolve(&node) {
                Ok(Object::Dict(d)) => d,
                Ok(other) => {
                    notes.push(format!(
                        "{what}: a tree node is a {}; skipped",
                        other.kind()
                    ));
                    continue;
                }
                Err(e) => {
                    notes.push(format!("{what}: a tree node is unreadable ({e}); skipped"));
                    continue;
                }
            };
            if let Ok(Some(Object::Array(pairs))) = self.get(&dict, leaves) {
                for pair in pairs.chunks(2) {
                    match (
                        self.resolve(&pair[0]).ok().as_ref().and_then(&key),
                        pair.get(1),
                    ) {
                        (Some(k), Some(v)) => out.push((k, v.clone())),
                        _ => notes.push(format!("{what}: a key without its value; skipped")),
                    }
                }
            }
            if let Ok(Some(Object::Array(kids))) = self.get(&dict, b"Kids") {
                // Pushed last first, so the leftmost kid is walked first and keys come out in tree order.
                stack.extend(kids.into_iter().rev().map(|k| (k, depth + 1)));
            }
        }
        out
    }
}
