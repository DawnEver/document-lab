//! The standard security handler (ISO 32000-2 section 7.6): the file key from a password, and each object's
//! strings and streams decrypted with it.
//!
//! Revisions 2-6: `RC4` (40-128 bit), AES-128 (`AESV2`) and AES-256 (`AESV3`), with crypt filters. The primitives
//! are the `RustCrypto` crates' (MD5, SHA-2, AES-CBC); `RC4` and everything PDF-specific are here. A file opens with
//! its user password or its owner password; a file meant to be read by anyone has the empty user password.

use aes::cipher::block_padding::NoPadding;
use aes::cipher::{BlockModeDecrypt, BlockModeEncrypt, KeyIvInit};
use md5::{Digest, Md5};
use sha2::{Sha256, Sha384, Sha512};

use crate::error::{Error, Kind, Result};
use crate::object::{Dict, ObjRef, Object};

/// The 32-byte padding string a password is completed with (Algorithm 2, step a).
const PAD: [u8; 32] = [
    0x28, 0xBF, 0x4E, 0x5E, 0x4E, 0x75, 0x8A, 0x41, 0x64, 0x00, 0x4E, 0x56, 0xFF, 0xFA, 0x01, 0x08,
    0x2E, 0x2E, 0x00, 0xB6, 0xD0, 0x68, 0x3E, 0x80, 0x2F, 0x0C, 0xA9, 0xFE, 0x64, 0x53, 0x69, 0x7A,
];

/// How a crypt filter encrypts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Method {
    /// Not encrypted (`/Identity`, `/None`).
    Identity,
    /// RC4 with the per-object key.
    Rc4,
    /// AES-128-CBC with the per-object key (`AESV2`).
    Aes128,
    /// AES-256-CBC with the file key itself (`AESV3`).
    Aes256,
}

/// A file's decryption: its key and how strings, streams and the metadata stream are encrypted.
#[derive(Debug, Clone)]
pub struct Decryptor {
    key: Vec<u8>,
    strings: Method,
    streams: Method,
    metadata: bool,
}

impl Decryptor {
    /// The decryptor of a file whose trailer names `encrypt` (resolved), opened with `password` (empty: none).
    ///
    /// # Errors
    ///
    /// Refuses a handler other than `/Standard`, a revision or method this reader does not know, and a password
    /// that opens neither the user's nor the owner's access.
    pub fn new(encrypt: &Dict, first_id: &[u8], password: &[u8]) -> Result<Self> {
        let filter = encrypt
            .get(b"Filter")
            .and_then(Object::as_name)
            .unwrap_or(b"");
        if filter != b"Standard" {
            return Err(Error::unsupported(format!(
                "the file is encrypted by the /{} security handler; only /Standard (password) is read",
                String::from_utf8_lossy(filter)
            )));
        }
        let int = |key: &[u8]| encrypt.get(key).and_then(Object::as_integer);
        let bytes = |key: &[u8]| match encrypt.get(key) {
            Some(Object::String(s)) => Ok(s.clone()),
            _ => Err(Error::malformed(format!(
                "the encryption dictionary has no /{} string",
                String::from_utf8_lossy(key)
            ))),
        };
        let version = int(b"V").unwrap_or(0);
        let revision =
            int(b"R").ok_or_else(|| Error::malformed("the encryption dictionary has no /R"))?;
        let metadata = !matches!(encrypt.get(b"EncryptMetadata"), Some(Object::Bool(false)));
        let (strings, streams) = match version {
            1 | 2 => (Method::Rc4, Method::Rc4),
            4 | 5 => (
                crypt_filter(encrypt, b"StrF")?,
                crypt_filter(encrypt, b"StmF")?,
            ),
            _ => {
                return Err(Error::unsupported(format!(
                    "encryption /V {version} is not read"
                )));
            }
        };
        let (owner, user) = (bytes(b"O")?, bytes(b"U")?);
        let key = match revision {
            2..=4 => {
                let length = if revision == 2 {
                    5
                } else {
                    usize::try_from(int(b"Length").unwrap_or(40) / 8).unwrap_or(5)
                };
                if !(5..=16).contains(&length) {
                    return Err(Error::malformed(format!(
                        "an RC4 key of {length} bytes is out of range"
                    )));
                }
                let permissions = i32::try_from(int(b"P").unwrap_or(-1)).unwrap_or(-1);
                let legacy = Legacy {
                    revision,
                    length,
                    owner: &owner,
                    user: &user,
                    permissions,
                    id: first_id,
                    metadata,
                };
                legacy
                    .user_key(password)
                    .or_else(|| legacy.owner_key(password))
            }
            5 | 6 => {
                let wrapped = (bytes(b"OE")?, bytes(b"UE")?); // spelling-ignore-line: the PDF names
                aes256_key(
                    revision,
                    password,
                    (&owner, &user),
                    (&wrapped.0, &wrapped.1),
                )
            }
            _ => {
                return Err(Error::unsupported(format!(
                    "security handler revision {revision} is not read"
                )));
            }
        };
        let key = key.ok_or_else(|| {
            Error::new(
                Kind::Password,
                "the password opens neither the user's nor the owner's access",
            )
        })?;
        Ok(Self {
            key,
            strings,
            streams,
            metadata,
        })
    }

    /// Decrypt every string in `value` and, when it is a stream, its data -- `value` being object `id`.
    ///
    /// # Errors
    ///
    /// Refuses AES data whose length is not whole 16-byte blocks after its IV.
    pub fn object(&self, id: ObjRef, value: &mut Object) -> Result<()> {
        if let Object::Stream(s) = value {
            let kind = s.dict.get(b"Type").and_then(Object::as_name);
            // A cross-reference stream is never encrypted; nor the metadata stream when the file says so.
            let skip = kind == Some(b"XRef")
                || (kind == Some(b"Metadata") && !self.metadata)
                || own_identity(&s.dict);
            if !skip {
                s.raw = self.decrypt(self.streams, id, &s.raw)?;
            }
        }
        self.strings_in(id, value, 0)
    }

    fn strings_in(&self, id: ObjRef, value: &mut Object, depth: usize) -> Result<()> {
        if depth > 256 {
            return Err(Error::malformed("objects nested too deep"));
        }
        match value {
            Object::String(s) => *s = self.decrypt(self.strings, id, s)?,
            Object::Array(items) => {
                for item in items {
                    self.strings_in(id, item, depth + 1)?;
                }
            }
            Object::Dict(d) => self.strings_in_dict(id, d, depth)?,
            Object::Stream(s) => self.strings_in_dict(id, &mut s.dict, depth)?,
            _ => {}
        }
        Ok(())
    }

    fn strings_in_dict(&self, id: ObjRef, dict: &mut Dict, depth: usize) -> Result<()> {
        for value in dict.values_mut() {
            self.strings_in(id, value, depth + 1)?;
        }
        Ok(())
    }

    fn decrypt(&self, method: Method, id: ObjRef, data: &[u8]) -> Result<Vec<u8>> {
        match method {
            Method::Identity => Ok(data.to_vec()),
            Method::Rc4 => Ok(rc4(&self.object_key(id, false), data)),
            Method::Aes128 => aes_cbc(&self.object_key(id, true), data),
            Method::Aes256 => aes_cbc(&self.key, data),
        }
    }

    /// Algorithm 1: the file key extended by the object number and generation (and `sAlT` for AES).
    fn object_key(&self, id: ObjRef, aes: bool) -> Vec<u8> {
        let mut h = Md5::new();
        h.update(&self.key);
        h.update(&id.num.to_le_bytes()[..3]);
        h.update(id.generation.to_le_bytes());
        if aes {
            h.update(b"sAlT");
        }
        h.finalize()[..(self.key.len() + 5).min(16)].to_vec()
    }
}

/// Whether a stream names the `/Identity` crypt filter for itself (its data then is not encrypted).
fn own_identity(dict: &Dict) -> bool {
    let named = |o: &Object| o.as_name() == Some(b"Crypt");
    let filters: Vec<&Object> = match dict.get(b"Filter") {
        Some(Object::Array(a)) => a.iter().collect(),
        Some(one) => vec![one],
        None => return false,
    };
    let params = dict.get(b"DecodeParms"); // spelling-ignore-line: the PDF name
    filters.iter().any(|f| named(f))
        && match params {
            Some(Object::Dict(p)) => p
                .get(b"Name")
                .and_then(Object::as_name)
                .is_none_or(|n| n == b"Identity"),
            _ => true,
        }
}

/// The method of the crypt filter `/StmF` or `/StrF` names (absent: `/Identity`).
fn crypt_filter(encrypt: &Dict, which: &[u8]) -> Result<Method> {
    let name = encrypt
        .get(which)
        .and_then(Object::as_name)
        .unwrap_or(b"Identity");
    if name == b"Identity" {
        return Ok(Method::Identity);
    }
    let filter = encrypt
        .get(b"CF")
        .and_then(Object::as_dict)
        .and_then(|cf| cf.get(name))
        .and_then(Object::as_dict)
        .ok_or_else(|| {
            Error::malformed(format!(
                "the crypt filter /{} is not defined",
                String::from_utf8_lossy(name)
            ))
        })?;
    match filter
        .get(b"CFM")
        .and_then(Object::as_name)
        .unwrap_or(b"None")
    {
        b"None" => Ok(Method::Identity),
        b"V2" => Ok(Method::Rc4),
        b"AESV2" => Ok(Method::Aes128),
        b"AESV3" => Ok(Method::Aes256),
        other => Err(Error::unsupported(format!(
            "crypt filter method /{} is not read",
            String::from_utf8_lossy(other)
        ))),
    }
}

/// The RC4-era handler's values (revisions 2-4).
struct Legacy<'a> {
    revision: i64,
    length: usize,
    owner: &'a [u8],
    user: &'a [u8],
    permissions: i32,
    id: &'a [u8],
    metadata: bool,
}

impl Legacy<'_> {
    /// Algorithm 2: the file key `password` gives.
    fn key(&self, password: &[u8]) -> Vec<u8> {
        let mut h = Md5::new();
        h.update(padded(password));
        h.update(&self.owner[..self.owner.len().min(32)]);
        h.update(self.permissions.to_le_bytes());
        h.update(self.id);
        if self.revision >= 4 && !self.metadata {
            h.update([0xFF; 4]);
        }
        let mut key = h.finalize()[..self.length].to_vec();
        if self.revision >= 3 {
            for _ in 0..50 {
                key = Md5::digest(&key)[..self.length].to_vec();
            }
        }
        key
    }

    /// Algorithms 4-6: the key, when `password` is the user password.
    fn user_key(&self, password: &[u8]) -> Option<Vec<u8>> {
        let key = self.key(password);
        let opens = if self.revision == 2 {
            rc4(&key, &PAD) == self.user.get(..32)?
        } else {
            let mut h = Md5::new();
            h.update(PAD);
            h.update(self.id);
            let mut check = rc4(&key, &h.finalize());
            for i in 1..=19u8 {
                check = rc4(&xor(&key, i), &check);
            }
            check[..16] == *self.user.get(..16)?
        };
        opens.then_some(key)
    }

    /// Algorithm 7: the key, when `password` is the owner password (it decrypts `/O` to the user password).
    fn owner_key(&self, password: &[u8]) -> Option<Vec<u8>> {
        let mut digest = Md5::digest(padded(password)).to_vec();
        if self.revision >= 3 {
            for _ in 0..50 {
                digest = Md5::digest(&digest).to_vec();
            }
        }
        let key = &digest[..self.length];
        let mut user = self.owner.get(..32)?.to_vec();
        if self.revision == 2 {
            user = rc4(key, &user);
        } else {
            for i in (0..=19u8).rev() {
                user = rc4(&xor(key, i), &user);
            }
        }
        self.user_key(&user)
    }
}

/// A password truncated or completed with [`PAD`] to 32 bytes.
fn padded(password: &[u8]) -> [u8; 32] {
    let mut out = PAD;
    let n = password.len().min(32);
    out[..n].copy_from_slice(&password[..n]);
    out[n..].copy_from_slice(&PAD[..32 - n]);
    out
}

fn xor(key: &[u8], i: u8) -> Vec<u8> {
    key.iter().map(|b| b ^ i).collect()
}

/// Algorithms 2.A and 2.B (revisions 5 and 6): the 32-byte file key, from the owner password or the user's.
///
/// `entries` are `/O` and `/U`; `wrapped` the file key wrapped for the owner and for the user.
fn aes256_key(
    revision: i64,
    password: &[u8],
    entries: (&[u8], &[u8]),
    wrapped: (&[u8], &[u8]),
) -> Option<Vec<u8>> {
    let password = &password[..password.len().min(127)];
    let (owner, user) = (entries.0.get(..48)?, entries.1.get(..48)?);
    let hash = |salt: &[u8], udata: &[u8]| {
        if revision == 5 {
            Sha256::digest([password, salt, udata].concat()).to_vec()
        } else {
            hash_2b(password, salt, udata)
        }
    };
    let unwrap = |wrapping: &[u8], wrapped: &[u8]| -> Option<Vec<u8>> {
        let mut buf = wrapped.get(..32)?.to_vec();
        let dec = cbc::Decryptor::<aes::Aes256>::new_from_slices(wrapping, &[0u8; 16]).ok()?;
        dec.decrypt_padded::<NoPadding>(&mut buf).ok()?;
        Some(buf)
    };
    if hash(&owner[32..40], user) == owner[..32] {
        return unwrap(&hash(&owner[40..48], user), wrapped.0);
    }
    if hash(&user[32..40], &[]) == user[..32] {
        return unwrap(&hash(&user[40..48], &[]), wrapped.1);
    }
    None
}

/// Algorithm 2.B: revision 6's hash, rounds of AES-128-CBC and SHA-256/384/512 chosen by the data.
fn hash_2b(password: &[u8], salt: &[u8], udata: &[u8]) -> Vec<u8> {
    let mut k = Sha256::digest([password, salt, udata].concat()).to_vec();
    let mut round = 0u32;
    loop {
        let block = [password, &k, udata].concat();
        let mut e = block.repeat(64);
        let len = e.len();
        let enc = cbc::Encryptor::<aes::Aes128>::new_from_slices(&k[..16], &k[16..32])
            .expect("16-byte key and IV");
        enc.encrypt_padded::<NoPadding>(&mut e, len)
            .expect("64 copies are whole blocks");
        let selector = e[..16].iter().map(|&b| u32::from(b)).sum::<u32>() % 3;
        k = match selector {
            0 => Sha256::digest(&e).to_vec(),
            1 => Sha384::digest(&e).to_vec(),
            _ => Sha512::digest(&e).to_vec(),
        };
        round += 1;
        if round >= 64 && u32::from(*e.last().expect("not empty")) <= round - 32 {
            return k[..32].to_vec();
        }
    }
}

/// `RC4`: `data` combined (exclusive or) with the keystream.
fn rc4(key: &[u8], data: &[u8]) -> Vec<u8> {
    let mut s: [u8; 256] = std::array::from_fn(|i| i.to_le_bytes()[0]);
    let mut j = 0u8;
    for i in 0..256 {
        j = j.wrapping_add(s[i]).wrapping_add(key[i % key.len()]);
        s.swap(i, usize::from(j));
    }
    let (mut i, mut j) = (0u8, 0u8);
    data.iter()
        .map(|&b| {
            i = i.wrapping_add(1);
            j = j.wrapping_add(s[usize::from(i)]);
            s.swap(usize::from(i), usize::from(j));
            b ^ s[usize::from(s[usize::from(i)].wrapping_add(s[usize::from(j)]))]
        })
        .collect()
}

/// AES-CBC with a 16- or 32-byte key: the first 16 bytes are the IV; the PKCS#7 padding is removed (a
/// malformed one is kept as data).
fn aes_cbc(key: &[u8], data: &[u8]) -> Result<Vec<u8>> {
    if data.is_empty() {
        return Ok(Vec::new());
    }
    if data.len() < 16 || !data.len().is_multiple_of(16) {
        return Err(Error::malformed(format!(
            "AES data of {} bytes is not an IV and whole blocks",
            data.len()
        )));
    }
    let (iv, body) = data.split_at(16);
    let mut buf = body.to_vec();
    let whole = |_| Error::malformed("AES data is not whole blocks");
    let wrong_key = |_| Error::malformed("an AES key is 16 or 32 bytes");
    if key.len() == 16 {
        let dec = cbc::Decryptor::<aes::Aes128>::new_from_slices(key, iv).map_err(wrong_key)?;
        dec.decrypt_padded::<NoPadding>(&mut buf).map_err(whole)?;
    } else {
        let dec = cbc::Decryptor::<aes::Aes256>::new_from_slices(key, iv).map_err(wrong_key)?;
        dec.decrypt_padded::<NoPadding>(&mut buf).map_err(whole)?;
    }
    let pad = usize::from(*buf.last().unwrap_or(&0));
    if (1..=16).contains(&pad)
        && pad <= buf.len()
        && buf[buf.len() - pad..]
            .iter()
            .all(|&b| usize::from(b) == pad)
    {
        buf.truncate(buf.len() - pad);
    }
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rc4_matches_its_published_vectors() {
        // The "Key"/"Plaintext" and "Wiki"/"pedia" test vectors.
        assert_eq!(
            rc4(b"Key", b"Plaintext"),
            [0xBB, 0xF3, 0x16, 0xE8, 0xD9, 0x40, 0xAF, 0x0A, 0xD3]
        );
        assert_eq!(rc4(b"Wiki", b"pedia"), [0x10, 0x21, 0xBF, 0x04, 0x20]);
    }

    #[test]
    fn a_password_is_completed_with_the_padding_string() {
        let p = padded(b"ab");
        assert_eq!(&p[..2], b"ab");
        assert_eq!(&p[2..], &PAD[..30]);
        assert_eq!(padded(&[7; 40]), [7; 32]);
    }
}
