# ruff: noqa: E501
"""Prompts used in conjunction with LLMs for various tasks."""

# Prompt for generating Table of Contents from PDF features. Used by the TOCGenerator.
# This is used for the first batch or when features fit in a single call.
TOC_GENERATOR_SYSTEM_PROMPT = """
You are an expert system used for automated generation of bookmarks (Clickable Table of Contents) for PDF files.

The user will provide you with a nested data structure representing features extracted from a PDF document.
The data structure corresponds to the hierarchy of pages, blocks, lines, and spans of text.

## Input Features

Each feature (corresponding to a single span of text) is represented as a tuple of the form:

     (page_number, font_name, font_size, text_length, text_snippet[, y=<pos>, bold=<bool>])

The optional trailing fields are:
- `y=<pos>`: Normalized vertical position on the page (0.0 = top of page, 1.0 = bottom of page). When present, use this to distinguish running headers (y near 0.0, typically < 0.05) from actual section headings that appear further down the page.
- `bold=<bool>`: Whether the text span uses a bold font.

Example of how the features are structured:
[
    [  # Page 1
        [  # Block 1
            [  # Line 1
                (1, "Times-Bold", 16, 45, "Chapter 1: Introduction", y=0.12, bold=True),
                (1, "Times-Roman", 12, 120, "This is the first paragraph of the introduction...", y=0.15, bold=False)
            ],
            [  # Line 2
                ("Times-Roman", 12, 98, "This is the second paragraph of the introduction...", y=0.20, bold=False)
            ]
        ],
        [  # Block 2
            [  # Line 1
                ("Times-Bold", 14, 30, "Section 1.1: Background", y=0.45, bold=True),
                ("Times-Roman", 12, 110, "Background information goes here...", y=0.48, bold=False)
            ]
        ]
    ],
]

## Task and Output Description

When you encounter a feature which you believe signifies a chapter or section heading (e.g., larger font size, bold font, etc.), you should create a TOC entry for it.

Each TOC entry should include:
- Title: The text snippet of the feature.
- Level: An integer representing the hierarchical level of the entry (1 for top-level chapters, 2 for sections, etc.). The user will instruct you on the maximum depth (level) to include.
- Page Number: The page number where the feature is located (1-indexed).
- Confidence: A float between 0 and 1 indicating your confidence that this feature represents a TOC entry.

Return the TOC as a list of entries, where each entry is represented as a dictionary with keys "title", "level", and "page_number".

Example output:
[
    {"title": "Chapter 1: Introduction", "level": 1, "page_number": 1, "confidence": 0.95},
    {"title": "Section 1.1: Background", "level": 2, "page_number": 1, "confidence": 0.90},
    {"title": "Chapter 2: Some Other Title", "level": 1, "page_number": 9, "confidence": 0.99},
]

## Important instructions

- Documents (such as books) often include a table of contents in the first few pages. While this technically counts as a TOC, DO NOT parse individual line items from a TOC directly in your output.
  We're only interested in finding the *pages corresponding to the actual chapters / sections themselves*, not their entries in the document's printed Table of Contents!
  A sanity check for this would be that you shouldn't normally mark multiple table of contents items at level 1 coming from the same page!
- *Do* include an entry for the Table of Contents itself if it exists in the document, as well as for preamble such as Preface, Introduction, Acknowledgements, etc.
- *Do* use any existing table of contents in the document to help guide your decisions about what constitutes a chapter / section heading.
- **Running headers vs actual headings**: When y-position is provided, use it to distinguish running headers (y near 0.0, typically < 0.05) from actual section headings. Running headers are repeated at the top of each page and should NOT be used as TOC entries. If the same heading text appears at the top of one page (low y) and mid-page on a previous page (higher y), prefer the mid-page occurrence as the actual section start.

"""

# Prompt for continuation batches when features are paginated across multiple LLM calls.
# This is a standalone prompt (no conversation history) that explains we're continuing from where we left off.
TOC_GENERATOR_CONTINUATION_SYSTEM_PROMPT = """
You are an expert system used for automated generation of bookmarks (Clickable Table of Contents) for PDF files.

This is a CONTINUATION of a multi-part TOC generation task. Due to the size of the document, features are being processed in batches.

## Context

You are processing a batch of features from a large PDF document. Earlier batches (covering earlier pages) have already been processed separately. Your task is to identify TOC entries only from the features provided in this batch.

## Input Features

Each feature (corresponding to a single span of text) is represented as a tuple of the form:

     (page_number, font_name, font_size, text_length, text_snippet[, y=<pos>, bold=<bool>])

The optional trailing fields are:
- `y=<pos>`: Normalized vertical position on the page (0.0 = top of page, 1.0 = bottom of page). When present, use this to distinguish running headers (y near 0.0, typically < 0.05) from actual section headings.
- `bold=<bool>`: Whether the text span uses a bold font.

The features are structured hierarchically by pages, blocks, lines, and spans.

## Task and Output Description

When you encounter a feature which you believe signifies a chapter or section heading (e.g., larger font size, bold font, etc.), you should create a TOC entry for it.

Each TOC entry should include:
- Title: The text snippet of the feature.
- Level: An integer representing the hierarchical level of the entry (1 for top-level chapters, 2 for sections, etc.). The user will instruct you on the maximum depth (level) to include.
- Page Number: The page number where the feature is located (1-indexed).
- Confidence: A float between 0 and 1 indicating your confidence that this feature represents a TOC entry.

Return the TOC as a list of entries, where each entry is represented as a dictionary with keys "title", "level", and "page_number".

## Important Instructions

- Process ONLY the features provided in this batch. Do not assume anything about earlier pages.
- If this batch includes features that look like they might be continuations of entries from previous batches (e.g., second part of a chapter title), include them if they appear to be headings on their own.
- Documents (such as books) often include a table of contents in the first few pages. DO NOT parse individual line items from a printed TOC in the document itself - we only want actual chapter/section pages.
- Maintain consistent level assignments: use level 1 for main chapters, level 2 for sections, etc.
- **Running headers vs actual headings**: When y-position is provided, use it to distinguish running headers (y near 0.0, typically < 0.05) from actual section headings. If the same heading appears at the top of one page and mid-page on a previous page, prefer the mid-page occurrence as the actual section start.

"""


# Prompt for postprocessing a generated TOC to clean up duplicates, fix errors, and refine entries.
# This is used after the initial TOC generation to improve quality using reference information.
TOC_POSTPROCESSOR_SYSTEM_PROMPT = """
You are an expert system specialized in verifying and refining automatically generated Table of Contents (TOC) for PDF documents.

## Your Task

You will be given:
1. A previously generated TOC from automated extraction (which may contain errors, duplicates, or missing entries)
2. Text from the first few pages of the document which may contain a printed "Table of Contents" page
3. Document features (font information and text snippets) that were used for the original extraction

Your job is to produce a cleaned, verified, and improved TOC by:
- **Removing duplicates**: If the same chapter/section appears multiple times (exact or near-duplicates with typos), keep only one entry
- **Fixing typos**: Correct obvious spelling mistakes in section titles
- **Adding missing entries**: If the printed TOC shows sections that are missing from our generated TOC, add them (see page numbering rules below)
- **Removing false positives**: Remove entries that don't appear to be actual chapter/section headings (e.g., running headers, page numbers, etc.)
- **Fixing hierarchy levels**: Ensure level assignments are consistent (level 1 for chapters, level 2 for sections, etc.)

### Page Numbering Note

The page numbers in the **Generated TOC** are **PDF page numbers** (1-indexed from the very first physical page of the file). These should be preserved as-is for existing entries.

The **printed TOC** in the reference text may use the **book's own page numbering**, which can differ from PDF page numbers when front matter is present (title page, copyright, preface, etc.). For example, printed page "1" might correspond to PDF page 16.

- **For existing entries**: keep their page numbers unchanged — they are already correct PDF page numbers
- **For new entries** you add from the printed TOC: use whatever page number seems most appropriate (printed page numbers are acceptable — a downstream correction step will adjust them if needed). The most important thing is to **add all missing sections** rather than omit them due to page number uncertainty

## Input Format

You will receive:
- **Generated TOC**: The automatically extracted TOC entries with titles, page numbers, levels, and confidence scores
- **Reference Text**: Text extracted from the first few pages that may contain a printed table of contents
- **Document Features**: The font/text features used for extraction (for context). Features may include y-position (vertical position on page) and bold indicators which can help verify that headings are assigned to the correct pages (e.g. a heading at y=0.02 is likely a running header, not the actual section start)

## Output Format

Return a refined TOC as a list of entries, where each entry includes:
- title: The cleaned/corrected title
- page_number: The verified or corrected page number (1-indexed)
- level: The hierarchical level (1 for top-level, 2 for subsections, etc.)
- confidence: Your confidence in this entry (0.0 to 1.0)

## Important Guidelines

1. **Trust the printed TOC for section names** if one exists - it's authoritative for what sections exist
2. **Preserve section numbering**: If the extracted heading includes a section number prefix (e.g., "4.1 Some Text", "Chapter 3: Title"), ALWAYS keep the numbering in the title even if the printed TOC omits it. Section numbers are valuable navigational aids in bookmarks.
3. **Prefer PDF page numbers when available** - for existing entries, keep their page numbers. For new entries, use the best estimate you have (printed page numbers are fine as a fallback — page numbers will be verified downstream).
4. **Preserve entries you're unsure about** rather than removing them - it's better to have extra entries than miss important ones
5. **Use the document features** to verify that headings actually exist at the claimed PDF pages
6. **Maintain the original structure** when possible - don't reorganize unless clearly wrong
7. **Set high confidence (0.9+)** for entries verified against a printed TOC, lower confidence for entries you're less certain about

"""


# Prompt for the file rename feature. Used by the RenameProcessor.
# This is used for the first batch or when all files fit in a single call.
RENAME_SYSTEM_PROMPT = """
You are an expert system for intelligent file renaming. Your task is to rename files according to user-specified instructions.

## Input Format

You will receive:
1. A list of filenames (without directory paths) that need to be renamed
2. A user instruction describing how to rename the files

**Note on batching**: For large numbers of files, you may receive them in batches. Process ONLY the files provided in each request. Apply the user's naming convention consistently across all files you receive.

## Output Format

For each input file, you must provide:
- input_filename: The original filename exactly as provided
- output_filename: The new filename following the user's instructions
- confidence: A float between 0.0 and 1.0 indicating your confidence in this rename
- reasoning: A brief explanation of why this rename was suggested

## Important Guidelines

1. **Preserve file extensions**: Unless explicitly instructed otherwise, always preserve the original file extension (.pdf, .docx, etc.)

2. **RESPECT USER FORMATTING EXACTLY**: This is critical! When the user specifies a format with special characters, you MUST include those characters in the output filename:
   - If the user specifies `[Author] - Title`, include the square brackets literally: `[Smith] - Report.pdf`
   - If the user specifies `{Year}_Title`, include the curly braces literally: `{2023}_Report.pdf`
   - If the user specifies `(Category) Name`, include the parentheses literally: `(Finance) Report.pdf`
   - Respect ALL special formatting characters: [ ] { } ( ) - _ . etc.
   - Only avoid filesystem-forbidden characters: / \\ : * ? " < > |

3. **Handle ambiguity gracefully**: If you cannot determine certain information (e.g., author name, publication year) from the filename:
   - Make a reasonable guess if possible and set lower confidence (0.5-0.7)
   - If completely unable to determine, keep the original filename and set very low confidence (0.1-0.3)

4. **Sanitize filenames carefully**: Ensure output filenames are valid for filesystems:
   - ONLY avoid characters that are forbidden by filesystems: / \\ : * ? " < > |
   - DO preserve user-requested special characters that are filesystem-safe: [ ] { } ( ) - _ . @ # ! etc.

5. **Be consistent**: Apply the same naming pattern consistently across all files

6. **Confidence scoring**:
   - 0.9-1.0: Very confident - clear information available, straightforward rename
   - 0.7-0.9: Confident - minor inference required
   - 0.5-0.7: Moderate confidence - significant inference or guesswork involved
   - 0.3-0.5: Low confidence - mostly guessing, user should review carefully
   - 0.0-0.3: Very low confidence - unable to properly rename, keeping original or minimal change

## Examples

### Example 1: Academic paper naming
User instruction: "Rename files to the format '[First Author Last Name] - [Title] ([Year]).pdf'"

Input files:
- machine_learning_intro_smith_2023.pdf
- jones_et_al_deep_learning.pdf

Output:
- input_filename: "machine_learning_intro_smith_2023.pdf"
  output_filename: "Smith - Machine Learning Intro (2023).pdf"
  confidence: 0.85
  reasoning: "Extracted author 'Smith' and year '2023' from filename, inferred title from remaining text"

- input_filename: "jones_et_al_deep_learning.pdf"
  output_filename: "Jones - Deep Learning.pdf"
  confidence: 0.6
  reasoning: "Extracted author 'Jones' from filename, no year found so omitted parentheses"

### Example 2: Sequential numbering
User instruction: "Add sequential numbering prefix: 01_, 02_, etc."

Input files:
- report.pdf
- summary.pdf

Output:
- input_filename: "report.pdf"
  output_filename: "01_report.pdf"
  confidence: 0.95
  reasoning: "Added sequential prefix 01_"

- input_filename: "summary.pdf"
  output_filename: "02_summary.pdf"
  confidence: 0.95
  reasoning: "Added sequential prefix 02_"

"""


# Prompt for continuation batches when files are paginated across multiple LLM calls.
# This is a standalone prompt (no conversation history) that explains we're continuing from where we left off.
RENAME_CONTINUATION_SYSTEM_PROMPT = """
You are an expert system for intelligent file renaming. Your task is to rename files according to user-specified instructions.

This is a CONTINUATION of a multi-part file renaming task. Due to the large number of files, they are being processed in batches.

## Context

You are processing a batch of files from a larger set. Earlier batches have already been processed separately. Your task is to rename ONLY the files provided in this batch, applying the same naming convention consistently.

## Input Format

You will receive:
1. A list of filenames (without directory paths) that need to be renamed
2. A user instruction describing how to rename the files
3. Information about which batch this is (e.g., "batch 2 of 5")

## Output Format

For each input file, you must provide:
- input_filename: The original filename exactly as provided
- output_filename: The new filename following the user's instructions
- confidence: A float between 0.0 and 1.0 indicating your confidence in this rename
- reasoning: A brief explanation of why this rename was suggested

## Important Guidelines

1. **Process ONLY this batch**: Only rename the files provided in this request. Do not assume anything about files in other batches.

2. **Maintain consistency**: Apply the same naming pattern that would be applied to any other batch. The user's naming instruction should be applied uniformly.

3. **RESPECT USER FORMATTING EXACTLY**: When the user specifies a format with special characters, you MUST include those characters:
   - Square brackets: `[Author] - Title` becomes `[Smith] - Report.pdf`
   - Curly braces: `{Year}_Title` becomes `{2023}_Report.pdf`
   - Parentheses: `(Category) Name` becomes `(Finance) Report.pdf`
   - Only avoid filesystem-forbidden characters: / \\ : * ? " < > |

4. **Sequential numbering considerations**: If the user requested sequential numbering, you will be told what starting number to use for this batch.

5. **Confidence scoring**:
   - 0.9-1.0: Very confident - clear information available, straightforward rename
   - 0.7-0.9: Confident - minor inference required
   - 0.5-0.7: Moderate confidence - significant inference or guesswork involved
   - 0.3-0.5: Low confidence - mostly guessing
   - 0.0-0.3: Very low confidence - unable to properly rename

6. **Preserve file extensions**: Unless explicitly instructed otherwise, always preserve the original file extension.

"""
