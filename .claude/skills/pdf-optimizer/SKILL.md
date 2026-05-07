---
name: pdf-optimizer
description: Diagnose and compress large PDFs while preserving text fidelity. Use whenever the user wants to "shrink", "compress", "optimize", or "reduce the size of" a PDF — especially scanned books, OCRed documents, or files over ~50MB. Also use when the user complains a PDF is "too big", "huge", "bloated", or "won't fit". The skill prevents the common mistake of reaching for `gs -dPDFSETTINGS=/ebook` (which silently *grows* MRC-encoded PDFs by destroying their layered structure) and instead picks the right tool based on what's actually inside the PDF.
---

# PDF Optimizer

Most "compress this PDF" advice on the internet recommends Ghostscript presets (`/ebook`, `/screen`, `/printer`). For modern OCRed PDFs from tools like ABBYY FineReader, those presets are *actively harmful*: they decode the carefully-tuned MRC (Mixed Raster Content) structure into a flat raster and re-encode it, which **often makes the file larger** and degrades text. The right approach depends entirely on what's inside the PDF — so always diagnose first.

## Step 1: Diagnose the PDF (always)

Before recommending anything, run these two commands and read the output:

```bash
pdfinfo "<file.pdf>"
pdfimages -list "<file.pdf>" 2>/dev/null | head -20
```

What to look for in `pdfinfo`:
- **Producer / Creator**: `ABBYY FineReader`, `Adobe Acrobat with OCR`, `tesseract` → likely scanned/OCRed
- `LaTeX`, `pdfTeX`, `Microsoft Word`, `Chrome` → born-digital
- **Pages**: gives a sense of expected size (~50-100 KB/page is normal for scanned text-heavy pages)

What to look for in `pdfimages -list`:
- A mix of `jbig2` (1-bit) **and** `jpeg`/`image` (color) entries per page → **MRC structure**, see Step 2A
- Only `jbig2` entries → simple OCR'd scan, see Step 2B
- Only `jpx`/`jpeg` entries with no JBIG2 → unprocessed scan, no MRC; see Step 2B (typically much larger savings possible)
- Almost no images, mostly text content streams → born-digital PDF, see Step 2C

To compute total bytes by encoding (helpful for prioritization):

```bash
pdfimages -list "<file.pdf>" 2>/dev/null | awk 'NR>2 {
  s = $(NF-1)
  if (s ~ /B$/) b = substr(s, 1, length(s)-1)
  else if (s ~ /K$/) b = substr(s, 1, length(s)-1) * 1024
  else if (s ~ /M$/) b = substr(s, 1, length(s)-1) * 1048576
  else b = s
  total += b; if ($9 ~ /jpe?g|jpx/) jpg += b; if ($9=="jbig2") jb += b
}
END { printf "JPEG: %.1fMB  JBIG2: %.1fMB  Total: %.1fMB\n", jpg/1048576, jb/1048576, total/1048576 }'
```

If JPEG bytes ≫ JBIG2 bytes, you're looking at a classic MRC PDF that's overspending on color background. This is the single biggest win — see Step 2A.

## Step 2A: MRC PDF (mixed JBIG2 + JPEG layers)

This is the case where the file looks like an ABBYY-style scan: each page has a 1-bit JBIG2 text mask plus one or two color JPEG layers (background paper, foreground ink color). Typical signature: a 918-page textbook coming in at 150-200 MB.

**The insight**: the JBIG2 mask is already optimal — that's what renders the text. The JPEG color layers are mostly a near-uniform paper-tan and near-uniform ink color, but ABBYY stored them as full-color JPEGs at quality ~80, which is wildly more than perception requires. Re-encode just those.

**The command**:

```bash
ocrmypdf --skip-text --optimize 2 -j 8 input.pdf output.pdf
```

What `--optimize 2` does:
- **Leaves JBIG2 streams byte-identical** → text rendering is bit-perfect
- Converts color JPEGs to palettized PNGs via `pngquant` (8-bit indexed; near-uniform paper tan compresses to almost nothing)
- Falls back to JPEG re-encode at quality 75 only when palette doesn't win
- Object stream cleanup for a few extra %

`--skip-text` tells ocrmypdf not to attempt new OCR (the document already has a text layer; new OCR would be wasteful and could replace the existing text layer with a worse one). `-j 8` parallelizes across 8 cores.

**Expected result**: 60-70% size reduction, zero text differences, pixel-identical body-page renders at 300 dpi.

**Why not `--optimize 3`?** It enables **lossy JBIG2 symbol substitution**, which can swap visually-similar glyphs (the famous Xerox-2013 bug: 6↔8 in scanned numbers). For prose this is usually fine; for **anything containing numbers, equations, code, or tables of figures, it's a no-go.** The marginal ~5% extra savings is not worth the risk of silent character substitution.

## Step 2B: Scanned PDF without MRC structure

If you see only JBIG2 (already-optimized scan) or only JPEG/JPX (raw scan, no MRC processing), the optimization path differs:

- **Only JBIG2, file already small (<50 KB/page)**: probably already optimal. Run qpdf for a free 5-10% lossless cleanup:
  ```bash
  qpdf --object-streams=generate --compress-streams=y --recompress-flate --compression-level=9 input.pdf output.pdf
  ```
- **Only JPEG/JPX, no text layer**: needs OCR to create the JBIG2 mask + MRC structure. Run full ocrmypdf:
  ```bash
  ocrmypdf --optimize 2 -j 8 input.pdf output.pdf
  ```
  (Note: no `--skip-text` here — we want to add OCR.) This can give 80%+ reductions on raw scans.

## Step 2C: Born-digital PDF

Files with embedded fonts and vector graphics (LaTeX output, Word exports, etc.) are usually already small. If they're large, the bloat is typically from:

- **Embedded high-resolution images** — diagnose with `pdfimages -list` looking for high `x-ppi`/`y-ppi` values (>300). Use ocrmypdf `--optimize 2` to recompress these.
- **Unsubsetted/duplicate fonts** — `qpdf` won't fix this, but `mutool clean -gggg -z` sometimes deduplicates streams. Lossless.
- **Just structural overhead** — `qpdf --object-streams=generate ...` for free 5-10%.

For born-digital PDFs, **never** use `gs /ebook` — it will rasterize vector content (graphs, equations) into blurry low-DPI images.

## Anti-patterns: tools that look helpful but hurt

| Tool / preset | Why it's wrong on MRC PDFs |
|---|---|
| `gs -dPDFSETTINGS=/printer` | Decodes pages into single rasters and re-encodes as JPEG. **Often makes MRC files larger.** Tested on a 157M textbook → 161M output. |
| `gs -dPDFSETTINGS=/ebook` | Same flattening problem, plus drops to 150 dpi (visible text degradation). Also extremely slow on long documents (10+ minutes for 900 pages). |
| `gs -dPDFSETTINGS=/screen` | Worst of both: 72 dpi flattening. Avoid except for "just need to email a preview." |
| Online "PDF compressor" sites | Almost all of them run gs with one of the above presets. Same outcome. |
| `mutool clean -gggg -z` | Lossless only — ceiling is ~5-10% savings. Acceptable as a fallback when ocrmypdf isn't available, but extremely slow on long documents (8+ minutes single-threaded for 900 pages). Prefer qpdf for the same lossless tier (4 seconds). |
| `qpdf --linearize` | Web optimization only, doesn't recompress images. Combine with the recommended qpdf invocation if linearization is desired. |

## Verifying fidelity after compression

When the user cares about preserving text quality (math, code, legal documents), verify before declaring success.

**Quick text-fidelity check**:

```bash
pdftotext -layout original.pdf /tmp/orig.txt
pdftotext -layout compressed.pdf /tmp/comp.txt
diff /tmp/orig.txt /tmp/comp.txt | grep -c '^[<>]'  # should be 0
wc -l /tmp/orig.txt
```

Zero diff lines across tens of thousands of lines is strong evidence text is preserved.

**Visual fidelity check** (renders pages and compares pixel-by-pixel):

```bash
# Pick a few representative pages spanning the document
for p in 1 50 100 250 400 600 800; do
  mutool draw -o /tmp/orig-p$p.png -r 300 -F png original.pdf $p 2>/dev/null
  mutool draw -o /tmp/comp-p$p.png -r 300 -F png compressed.pdf $p 2>/dev/null
  rmse=$(compare -metric RMSE /tmp/orig-p$p.png /tmp/comp-p$p.png /tmp/diff.png 2>&1 | awk -F'[()]' '{print $2}')
  printf "p%-4s RMSE=%s\n" "$p" "$rmse"
done
```

Interpretation:
- `RMSE=0` → pixel-identical (best case for body text on MRC PDFs with `-O2`)
- `RMSE < 0.01` → imperceptible (typical for cover pages and figure pages, where JPEG re-encoding noise is below 1 LSB at viewing resolution)
- `RMSE > 0.05` → visible degradation, probably should reject

**JBIG2-preserved check** (proves text mask wasn't touched):

```bash
pdfimages -list original.pdf | grep jbig2 | awk '{s=$(NF-1); print s}' | sort | md5
pdfimages -list compressed.pdf | grep jbig2 | awk '{s=$(NF-1); print s}' | sort | md5
# Same hash → JBIG2 sizes preserved (strong but not bulletproof signal of byte-identity)
```

## Tools and installation (macOS)

This skill relies on a small set of command-line tools. On macOS, install everything via Homebrew with one command:

```bash
brew install poppler ghostscript imagemagick qpdf mupdf-tools ocrmypdf jbig2enc pngquant
```

Most of these are usually already installed by other tooling (poppler ships with many TeX/PDF packages, ghostscript and imagemagick are common); the ones you most often need to add are `qpdf`, `mupdf-tools`, `ocrmypdf`, `jbig2enc`, and `pngquant`.

To check what's already present:

```bash
for t in pdfinfo pdfimages pdftotext gs qpdf mutool ocrmypdf jbig2 pngquant compare; do
  command -v $t >/dev/null && echo "✓ $t" || echo "✗ $t (missing)"
done
```

### What each tool is for

| Tool | Provided by | Used for |
|---|---|---|
| `pdfinfo` | `poppler` | **Diagnosis.** Producer/Creator metadata, page count, dimensions. |
| `pdfimages -list` | `poppler` | **Diagnosis.** Lists every embedded image with encoding (jpeg/jbig2/jpx/ccitt), DPI, and byte size. The single most informative command for picking a strategy. |
| `pdftotext -layout` | `poppler` | **Verification.** Extract text from both original and compressed; diff to confirm zero text loss. |
| `ocrmypdf` | `ocrmypdf` | **Primary optimizer.** With `--skip-text --optimize 2`, recompresses color JPEG layers via pngquant while leaving JBIG2 text masks byte-identical. Adds OCR if missing. Multi-process (`-j N`). |
| `jbig2` (jbig2enc) | `jbig2enc` | Backend that ocrmypdf calls when it does add or rebuild JBIG2 streams. Not invoked directly. Note: only ocrmypdf's `-O3` mode uses lossy JBIG2 — avoid for documents with numbers/math. |
| `pngquant` | `pngquant` | Backend that ocrmypdf calls to convert color JPEG layers to palettized PNGs. This is the tool doing the actual JPEG→tiny-PNG transformation that drives most of the savings on MRC PDFs. Not invoked directly. |
| `qpdf` | `qpdf` | **Lossless fallback.** Object-stream rewrite, recompress flate streams, generate compressed cross-references. Fast (~4s for a 157 MB book), ~5-10% savings ceiling. Only useful when image recompression is off the table. |
| `mutool` | `mupdf-tools` | **Verification rendering.** `mutool draw -r 300 -F png file.pdf N` rasterizes a single page deterministically — used for pixel-level fidelity checks. Also: `mutool clean -gggg -z` does lossless cleanup but is *very* slow on long documents (8+ minutes for 900 pages); prefer qpdf. |
| `gs` (Ghostscript) | `ghostscript` | **Mostly an anti-pattern for this skill.** Documented here so you can recognize what *not* to do. The popular `-dPDFSETTINGS=/ebook|printer|screen` presets flatten MRC structure and often grow MRC files. Only useful for born-digital PDFs with no image content, or to render single pages for diagnosis. |
| `compare` (ImageMagick) | `imagemagick` | **Verification.** `compare -metric RMSE a.png b.png diff.png` produces a pixel-difference score in [0,1]. RMSE=0 on body-text pages is the gold-standard proof that text rendering is unchanged. |

### Notes on the install

- `ocrmypdf` pulls in `tesseract` automatically as a dependency (used when adding new OCR; not used by `--skip-text`).
- `mupdf-tools` is the formula name; the binary is `mutool`.
- On Apple Silicon, all of these install cleanly to `/opt/homebrew/bin`. No Rosetta needed.
- These tools are also available on Linux (`apt install poppler-utils ghostscript imagemagick qpdf mupdf-tools ocrmypdf` plus `jbig2enc` and `pngquant` from your distro's repos) — the skill's commands are portable.

## Worked example

A 157 MB, 918-page numerical analysis textbook from ABBYY FineReader 15.

1. **Diagnose**:
   - `pdfinfo` → Producer: ABBYY FineReader PDF 15
   - `pdfimages -list` shows JBIG2 masks + JPEG layers per page
   - Bytes by encoding: JPEG 102 MB, JBIG2 29 MB → classic MRC, JPEGs dominate
2. **Optimize**:
   ```bash
   ocrmypdf --skip-text --optimize 2 -j 8 input.pdf output.pdf
   ```
   Runs in ~3 minutes.
3. **Result**: 58 MB (37% of original, 63% saved). JBIG2 unchanged at 29 MB; JPEGs reduced to 16.5 MB.
4. **Verify**: full-text diff = 0 lines across 50,915 lines; body pages render with RMSE = 0 at 300 dpi; cover page RMSE = 0.0025 (imperceptible).

This pattern — diagnose → identify the dominant byte category → optimize that category specifically — is the core of the skill. Don't skip the diagnosis step, even if the user just says "make it smaller."
