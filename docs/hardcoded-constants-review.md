# Hard-coded Constants in TOC Generation: Review and Alternatives

## Overview

The TOC generation pipeline in `pdfalive/processors/toc_generator.py` uses several hard-coded string patterns, dictionaries, and threshold constants. This document catalogs them, describes where they sit in the pipeline, and evaluates alternatives — particularly delegating more work to the LLM.

## Constants Inventory

### String/Pattern Matching

| Constant | Type | Purpose |
|----------|------|---------|
| `_FRONT_MATTER_TITLES` | `frozenset` of strings | Skip known front matter titles (Contents, Preface, Introduction, Foreword, Acknowledgements, etc.) when detecting the printed-to-PDF page offset |
| `_SECTION_NUMBER_PATTERN` | Compiled regex | Match section numbering prefixes: "1.", "1.2", "Chapter N", "Section N", "Part N", "Appendix N", Roman numerals ("XIV."), letter-spaced "C H A P T E R" |
| `_LETTERSPACED_PATTERN` | Compiled regex | Match ALL-CAPS letter-spaced text like "P R E F A C E" |
| `_ROMAN_NUMERAL_RE` | Regex sub-pattern | Match Roman numerals I through XXXIX (used inside `_SECTION_NUMBER_PATTERN`) |

### Numeric Thresholds

| Constant | Value | Purpose |
|----------|-------|---------|
| `_HEADING_MIN_LENGTH` | 3 | Minimum text length for a heading candidate |
| `_HEADING_MAX_LENGTH` | 200 | Maximum text length for a heading candidate |
| `_HEADING_FONT_SIZE_RATIO` | 1.15 | Font must be >= 1.15x body text size to be considered a heading |

## Where They Sit in the Pipeline

### Phase 0: Feature Extraction (pre-LLM)

**Used by:** `_is_heading_candidate()`, called from `_extract_features_sequential()` and `_extract_features_parallel()`

After extracting the first N blocks per page, remaining blocks are scanned for heading candidates. The constants decide which overflow spans are worth including as features sent to the LLM.

**Constants involved:**
- `_SECTION_NUMBER_PATTERN` — text-based heading detection
- `_LETTERSPACED_PATTERN` — text-based heading detection
- `_HEADING_MIN_LENGTH` / `_HEADING_MAX_LENGTH` — length bounds
- `_HEADING_FONT_SIZE_RATIO` — font-size-based heading detection

**Key point:** The LLM never sees spans that are filtered out here. This is a gate that determines what information reaches the LLM.

### Phase 2: Post-LLM Correction (after postprocessing)

**Used by:** `_detect_front_matter_offset()`, called from `_correct_postprocessed_page_numbers()`

After the postprocessor LLM refines the TOC, the correction code identifies the front matter offset by finding the first "real" chapter in the Phase 1 TOC. `_FRONT_MATTER_TITLES` determines which entries to skip.

**Constants involved:**
- `_FRONT_MATTER_TITLES` — title classification

**Key point:** This runs after both LLM passes. It's a deterministic heuristic trying to fix what the LLM may have gotten wrong.

## Brittleness Concerns

### `_FRONT_MATTER_TITLES` — High brittleness

- **English-only:** Won't match "Inhaltsverzeichnis", "Pr&eacute;face", "&Iacute;ndice", "Sommaire", etc.
- **Requires ongoing maintenance:** We already had to add "introduction" and switch from exact matching to prefix matching because "Acknowledgments for the English Edition" didn't match. Every book with unusual front matter labels is a potential failure.
- **Ambiguous entries:** "Introduction" can be either front matter or the first real chapter depending on the book. Currently treated as front matter.

### `_SECTION_NUMBER_PATTERN` — Moderate brittleness

- **English-centric named prefixes:** "Chapter", "Section", "Part", "Appendix" won't match "Cap&iacute;tulo", "Chapitre", "Kapitel", "Abschnitt", etc.
- **Roman numerals are universal:** The `_ROMAN_NUMERAL_RE` sub-pattern works across languages.
- **Letter-spacing is universal:** `_LETTERSPACED_PATTERN` works for any Latin-alphabet text.

### Numeric thresholds — Low brittleness

- `_HEADING_FONT_SIZE_RATIO` (1.15x) is a reasonable universal heuristic — larger font means heading regardless of language.
- `_HEADING_MIN/MAX_LENGTH` (3–200) are generous bounds unlikely to cause issues.

## Alternatives: Delegating to the LLM

### `_FRONT_MATTER_TITLES` — Strong candidate for removal

This is the most problematic constant and the most replaceable. Options:

1. **Remove offset detection entirely.** The "prompt-first" approach (telling the LLM to always output PDF page numbers with an explicit conversion formula) is already in place. If the LLM follows the prompt correctly, no downstream offset correction is needed. The correction code becomes a safety net, not the primary mechanism. We could simplify it to only restore matched entries' page numbers without trying to detect/apply offsets.

2. **Ask the LLM to identify the offset.** Include Phase 1 TOC entries in the postprocessor prompt and ask the LLM to determine the front matter offset itself. The LLM can understand "Contents", "Pr&eacute;face", "Einleitung" etc. without a hard-coded list.

3. **Use the LLM to classify entries.** Instead of a title list, ask the LLM in the postprocessor prompt: "Which of these entries are front matter vs. main content?" This is inherently multilingual.

**Recommendation:** Option 1 (simplify/remove) is the cleanest. The offset detection was added as a band-aid for the LLM outputting wrong page numbers. With clear prompts and the conversion formula, the LLM should handle this directly. The correction code should focus only on restoring matched entries' page numbers (which is language-neutral — it uses the Phase 1 page numbers, not title matching).

### `_SECTION_NUMBER_PATTERN` / `_LETTERSPACED_PATTERN` — Partial candidate

These are used in feature extraction to decide which spans to send to the LLM. The LLM can't replace a pre-LLM filter. Options:

1. **Relax the filter.** Send more spans to the LLM (increase `max_blocks_per_page` or reduce filtering strictness) and let the LLM decide what's a heading. Cost: more tokens per call.

2. **Keep font-based heuristics, drop text-based patterns.** The font size ratio and bold detection are language-neutral and effective. The regex patterns add English-specific assumptions on top. We could keep `_HEADING_FONT_SIZE_RATIO` and bold detection as the only heading candidate criteria, dropping the regex patterns.

3. **Keep as-is.** These patterns are a performance optimization (fewer tokens sent to the LLM). The LLM still makes the final heading decision — these just control which candidates it sees. Mis-classification here means a missed heading, but the LLM compensates from the blocks it does see.

**Recommendation:** Option 2 (keep font heuristics, drop text patterns) is a good balance. Font size and bold are universal signals. The regex patterns add marginal value at the cost of English-only assumptions.

### Numeric thresholds — Keep as-is

These are language-neutral and unlikely to cause issues. No action needed.

## Summary

| Constant | Brittleness | i18n impact | Recommendation |
|----------|-------------|-------------|----------------|
| `_FRONT_MATTER_TITLES` | High | Breaks for non-English | Remove or delegate to LLM |
| `_SECTION_NUMBER_PATTERN` | Moderate | Named prefixes are English-only | Drop text patterns, keep font heuristics |
| `_LETTERSPACED_PATTERN` | Low | Works for Latin alphabets | Could drop (low marginal value) |
| `_ROMAN_NUMERAL_RE` | Low | Universal | Could drop (part of section pattern) |
| `_HEADING_FONT_SIZE_RATIO` | Low | Universal | Keep |
| `_HEADING_MIN/MAX_LENGTH` | Low | Universal | Keep |
