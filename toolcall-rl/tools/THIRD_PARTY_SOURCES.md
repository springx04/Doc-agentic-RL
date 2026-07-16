# Third-party tool sources

External repositories were downloaded only into the temporary directory below
for inspection and interface extraction:

`C:\Users\30738\AppData\Local\Temp\openclaw_tool_sources`

No full third-party repository is vendored into this project.

## Sources checked

| Tool area | Source | Local temp path | Revision / status | Used for |
|---|---|---|---|---|
| PDF rendering | Local ROLL DocSeek PyMuPDF backend | `C:\Users\30738\Desktop\project\agent\ROLL\docseek\tools\pymupdf_backend.py` | local workspace | `render_page` and crop rendering pattern |
| OCR | RapidOCR | `...\RapidOCR` | `b64a46ffd4fc370e4e2c5a10c485cb26e40b8b0f` | `RapidOCR()` construction and `boxes/txts/scores` result normalization |
| Chart-to-table | Google Pix2Struct | `...\pix2struct` | `67e9f3080850d063c74d65f6336ac86fe817fb04` | DePlot/Pix2Struct model family reference |
| Chart-to-table | Hugging Face `google/deplot` model card | <https://huggingface.co/google/deplot> | inspected 2026-07-07 | `Pix2StructProcessor` + `Pix2StructForConditionalGeneration` call path |
| Table extraction | Camelot | `...\camelot` | `1a5275346732026223c709d0938a6d56f806b074` | `camelot.read_pdf(...).df` extraction path |
| Table extraction | pdfplumber | `...\pdfplumber` | downloaded from GitHub codeload stable zip | explicit `pdfplumber` engine path using `page.crop(...)` and `page.extract_tables()` |
| Document parsing/layout | Local Docling checkout | `C:\Users\30738\Desktop\project\agent\docling` | user-provided local source checkout | Reference for document/element/table JSON concepts only; no runtime import |
| Document parsing/layout | Docling official docs | <https://docling-project.github.io/docling/usage/> | inspected before local source became available | API and output concept cross-check |

## Download notes

Docling source download was attempted through:

- `git clone --depth 1 https://github.com/docling-project/docling.git`
- GitHub codeload zip
- `python -m pip download --no-deps docling`

All three failed in this environment with TLS/SSL EOF or handshake errors. The
user later provided a local checkout at
`C:\Users\30738\Desktop\project\agent\docling`. That checkout is treated as a
temporary/reference source only. Runtime tool code lives under
`toolcall-rl/tools/` and must not add the checkout to `sys.path` or import from it.
