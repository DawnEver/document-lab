"""Write the encrypted fixtures of `crates/pdf/tests/crypt.rs`: one synthetic page per encryption PyMuPDF writes.

Each holds the text ``secret text 42`` and the title ``Secret Title``; user password ``user``, owner ``owner``,
and an ``open-*`` twin anyone can open (empty user password). Run from this directory:
``python make_encrypted.py``.
"""

from pathlib import Path

import pymupdf

METHODS = {
    'rc4-40': pymupdf.PDF_ENCRYPT_RC4_40,
    'rc4-128': pymupdf.PDF_ENCRYPT_RC4_128,
    'aes-128': pymupdf.PDF_ENCRYPT_AES_128,
    'aes-256': pymupdf.PDF_ENCRYPT_AES_256,
}


def write(path: Path, method: int, user: str) -> None:
    """Write the page to ``path``, encrypted by ``method`` with user password ``user``."""
    with pymupdf.open() as doc:
        doc.new_page().insert_text((72, 72), 'secret text 42', fontname='helv')
        doc.set_metadata({'title': 'Secret Title'})
        doc.save(path, encryption=method, user_pw=user, owner_pw='owner', no_new_id=True, garbage=0, deflate=True)


if __name__ == '__main__':
    here = Path(__file__).parent
    for name, method in METHODS.items():
        write(here / f'{name}.pdf', method, 'user')
        write(here / f'open-{name}.pdf', method, '')
