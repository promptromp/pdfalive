---
name: pdf-combiner
description: Merge multiple PDFs into one in a user-specified order, then hand off to pdf-optimizer for compression. Use whenever the user wants to "merge", "combine", "concatenate", "join", or "stitch" PDFs together — including phone photos or scans of math homework, lecture notes, or graph-paper pages, individual chapter PDFs, or per-page exports that need to be assembled into a single document. Prevents the common mistake of using `gs` to merge (which silently re-encodes images and bloats scanned input), and shows the right combination of qpdf for ordered concatenation plus pdfunite/mutool fallbacks. Pairs with pdf-optimizer for the post-merge compression step.
---

# PDF Combiner

Merging PDFs sounds trivial — until you realize the popular Ghostscript one-liner re-encodes every page through a JPEG round-trip, which on scanned/photo input often *grows* the file and degrades grid lines and handwriting. The right tools for ordered, lossless merging are `qpdf` (preferred), `pdfunite`, and `mutool merge`. Optimization is a *separate* step handled by the [pdf-optimizer](../pdf-optimizer/SKILL.md) skill — never try to do both at once with `gs`.

## Step 1: Merge in user-specified order (lossless)

`qpdf` is the most flexible: it preserves bookmarks/metadata from the primary input, accepts page ranges per file, and is fast.

**Explicit ordering** (most common — user lists files in the order they want):

```bash
qpdf --empty --pages a.pdf b.pdf c.pdf -- merged.pdf
```

The `--empty` says "start from a blank document and append," which is what you want when no input file is special. The `--` terminates the page selection list.

**Numeric file ordering** (e.g. `page-001.pdf` ... `page-042.pdf`) — never rely on shell glob alone, which sorts lexically and breaks at `page-10.pdf`:

```bash
# zero-padded names — plain sort is fine
qpdf --empty --pages page-*.pdf -- merged.pdf

# unpadded names — use sort -V (version sort) to get numeric order
qpdf --empty --pages $(ls page-*.pdf | sort -V) -- merged.pdf
```

**From an order-list file** (one path per line, in desired order — useful for many files or arbitrary order):

```bash
mapfile -t files < order.txt
qpdf --empty --pages "${files[@]}" -- merged.pdf
```

**Pick page ranges per file** (e.g. drop a cover sheet, take only odd pages, reverse a duplex scan):

```bash
qpdf --empty --pages a.pdf 1-z b.pdf 2-z c.pdf 1,3,5 d.pdf z-1 -- merged.pdf
```

Range syntax: `1-z` = all pages, `2-z` = drop page 1, `z-1` = reverse, `1,3,5` = explicit list, `1-z:even` / `1-z:odd` = parity. See `qpdf --help=page-ranges`.

## Step 2: Optimize the merged file

Use the existing [pdf-optimizer](../pdf-optimizer/SKILL.md) skill — diagnose first, then apply the right tool. For the math-homework-on-graph-paper case (raw phone photos or scans, no text layer, large file), the typical recipe is:

```bash
ocrmypdf --optimize 2 --rotate-pages --deskew -j 8 merged.pdf final.pdf
```

What each flag does for graph-paper-style input:
- `--optimize 2`: builds MRC structure (1-bit JBIG2 for ink + palettized PNG for paper). Massive savings because the graph-paper grid is near-uniform color and palettizes to almost nothing.
- `--rotate-pages`: phone photos are often 90°/180° off; this auto-orients per page.
- `--deskew`: corrects small rotation from hand-held photos so the grid lines are straight.
- OCR runs by default and adds a searchable text layer for any printed parts; for purely handwritten pages tesseract output is mostly noise but the file-size impact is negligible. To skip OCR entirely, add `--tesseract-timeout 0`.

Do not pass `--optimize 3` here: it enables lossy JBIG2 symbol substitution, which can swap visually-similar digits — disastrous for math homework. See the pdf-optimizer skill's discussion of `-O2` vs `-O3`.

## Fallback merge tools

| Tool | When to use |
|---|---|
| `pdfunite a.pdf b.pdf c.pdf out.pdf` | Simple ordered merge. Drops bookmarks. Fine for plain scans where there's nothing structural to preserve. |
| `mutool merge -o out.pdf a.pdf b.pdf c.pdf` | More tolerant of slightly malformed input PDFs than qpdf. Also supports per-file page ranges: `mutool merge -o out.pdf a.pdf 1-3 b.pdf 5,7,9`. |
| Programmatic (PyMuPDF) | Use when you need conditional logic (filter pages by content, insert separators between docs, etc.). See snippet below. |

Programmatic merge (when shell isn't enough — e.g. you're building this into a script, or need to insert per-source TOC entries):

```python
import pymupdf

out = pymupdf.open()
for path in ["a.pdf", "b.pdf", "c.pdf"]:
    src = pymupdf.open(path)
    out.insert_pdf(src)
    src.close()
out.save("merged.pdf", garbage=4, deflate=True)
out.close()
```

`garbage=4` removes unreferenced objects, `deflate=True` recompresses streams — both lossless. Pair with `pdf-optimizer` afterward for the actual size win.

## Anti-patterns

| Approach | Why it's wrong |
|---|---|
| `gs -sDEVICE=pdfwrite -o out.pdf a.pdf b.pdf c.pdf` | Re-encodes images through JPEG even at default quality. On scanned/photo input the merged file is often **larger** than the sum of inputs, and grid lines / pencil strokes pick up ringing artifacts. Avoid for any merge involving raster content. |
| `gs -dPDFSETTINGS=/ebook` as a "merge + compress" one-liner | Compounds the above: same re-encode plus a forced 150 dpi downsample that visibly degrades handwriting. Always merge losslessly first, then optimize as a separate step. |
| Online "merge PDF" sites | Most run gs server-side with the presets above. Same outcome, plus you've uploaded potentially private homework to a third party. |
| `cat a.pdf b.pdf c.pdf > merged.pdf` | Not a joke — people try this. PDF is not a stream format; the output is a corrupt file. |
| Wildcards without `sort -V` for unpadded numeric names | `page-1.pdf page-10.pdf page-2.pdf ...` — your "merged" doc will be in lexical order, which is rarely what was intended. |

## Quick reference

| Want | Command |
|---|---|
| Merge a few files in given order | `qpdf --empty --pages a.pdf b.pdf c.pdf -- out.pdf` |
| Merge zero-padded numeric files | `qpdf --empty --pages page-*.pdf -- out.pdf` |
| Merge unpadded numeric files | `qpdf --empty --pages $(ls page-*.pdf \| sort -V) -- out.pdf` |
| Merge from an order-list file | `mapfile -t f < list.txt && qpdf --empty --pages "${f[@]}" -- out.pdf` |
| Drop pages while merging | `qpdf --empty --pages a.pdf 2-z b.pdf 1-3 -- out.pdf` |
| Reverse a doc while merging | `qpdf --empty --pages a.pdf z-1 -- out.pdf` |
| Optimize after merging (graph paper / scans) | `ocrmypdf --optimize 2 --rotate-pages --deskew -j 8 in.pdf out.pdf` |
| Verify page count | `pdfinfo merged.pdf \| grep ^Pages` |

## Tools

All required tools are already covered by the pdf-optimizer skill's install line:

```bash
brew install poppler ghostscript imagemagick qpdf mupdf-tools ocrmypdf jbig2enc pngquant
```

`qpdf` and `pdfunite` (from `poppler`) and `mutool` (from `mupdf-tools`) are the three merge backends; `ocrmypdf` handles the optimization step. `gs` is listed only so you can recognize the anti-pattern.

## Worked example: 30 phone photos of math homework on graph paper

User has `IMG_0001.jpg.pdf` ... `IMG_0030.jpg.pdf` (single-page PDFs, each ~4 MB from a 12 MP camera, ~120 MB total) and wants them combined chronologically.

1. **Merge** (numeric, zero-padded — a plain glob is fine):
   ```bash
   qpdf --empty --pages IMG_*.jpg.pdf -- merged.pdf
   ```
   Result: 120 MB, 30 pages. Takes ~1 second.

2. **Diagnose** (per pdf-optimizer skill):
   ```bash
   pdfimages -list merged.pdf | head -5
   ```
   Likely shows only `jpeg` or `jpx` entries, no `jbig2` — i.e. no MRC structure yet. This is the case where ocrmypdf gives the biggest wins.

3. **Optimize**:
   ```bash
   ocrmypdf --optimize 2 --rotate-pages --deskew -j 8 merged.pdf final.pdf
   ```
   Typical result: 12-25 MB (5-10× reduction), pages straightened, searchable for any printed text. Handwriting is preserved as the JBIG2 foreground mask.

4. **Verify** (per pdf-optimizer skill):
   ```bash
   for p in 1 15 30; do
     mutool draw -o /tmp/o-$p.png -r 300 -F png merged.pdf $p 2>/dev/null
     mutool draw -o /tmp/c-$p.png -r 300 -F png final.pdf $p 2>/dev/null
     compare -metric RMSE /tmp/o-$p.png /tmp/c-$p.png /tmp/d.png 2>&1 | awk -F'[()]' '{print "p'$p' RMSE="$2}'
   done
   ```
   Expect RMSE < 0.02 on body pages — imperceptible at viewing resolution.

The key pattern: **merge losslessly first, optimize as a separate step**. Never let one tool do both — you lose either control over ordering or fidelity over compression, usually both.
