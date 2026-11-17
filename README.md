# OCR pipeline

Local-first Python codebase for building one Markdown-and-images bundle per
document. Pipeline steps are isolated so that a step can consume saved outputs
from an earlier step and can be rerun independently.

## Requirements

- Python 3.11 or newer
- Install runtime dependencies with `python -m pip install -e .`
- Export the shared box spreadsheet as a UTF-8 CSV with a `box_code` column

## Batch folder layout

```powershell
F:\ocr\batch01\
├── raw\
│   └── B0012\
│       └── 01\
│           ├── B0012-01-00001.tif
│           └── ...
├── box_list.csv
├── batch_manifest.yaml
├── config.json
└── work\                     # Created by the pipeline
```

Commands receive the batch root (`F:\ocr\batch01`), not the `raw` directory.
The pipeline scans only `raw\`; the box list, `config.json`, and reports are
not treated as scans. Each unit must follow
`raw\<box-code>\<folder-or-book-number>\<scan-file>`, and the two directory
names must match the corresponding filename fields.

## Scan operator procedure

Before a production batch, verify scan resolution, grayscale/color mode,
TIFF output, duplex operation for folders, and the destination naming rule.
Disable scanner auto-enhancement, blank-page deletion, and automatic cropping;
the pipeline must preserve source scans as received.

For each batch, record the operator and scan setup in `batch_manifest.yaml`.
Use the shared box list for box codes, number each folder/book in scan order,
remove staples and protect damaged pages, and scan folders through the duplex
feeder or books page-by-page without forcing the spine. Review the first
operator batches page-by-page; after training, visually review at least one in
every ten pages and every page flagged by the scanner. If a feed, cut-off,
blur, skew, or missing-page problem is found, correct and rescan the entire
affected folder/book, then review that unit again. Reassemble physical records
in their original order after scanning.

## Step 1: check a batch

Without a manifest, each invocation checks one kind of scan unit: `folder`
(duplex, even page count) or `book`. With `batch_manifest.yaml`, the manifest
provides the type for each unit, so a batch can contain both. The box list
defaults to `<batch>\box_list.csv`; reports are written to
`<batch>\work\step-1-checks`.

```powershell
python -m ocr_pipeline check-batch `
  --batch "F:\ocr\batch01" `
  --manifest "F:\ocr\batch01\batch_manifest.yaml"
```

Use `--kind folder` or `--kind book` when no manifest is available. Optional
`--box-list` and `--output` arguments override the defaults.

The filename rule is `<box-code>-<unit-number>-<5-digit-page>.tif`, such as
`B0012-01-00001.tif`. The extension must be `.tif`; page numbering starts at
`00001`, with no gaps or repeats. Box codes are compared case-insensitively
with the CSV.

The output contains `batch-report.json` and a JSON report for each unit. A
failed unit must return to scanning and is skipped by Step 2. The command exits:

- `0` when every identified unit passes
- `1` when one or more units fail validation
- `2` when the inputs cannot be checked or reports cannot be written

Unattributed malformed scan filenames are recorded as batch-level errors and
fail all identified units. The optional YAML manifest describes the batch,
operator, scan setup and item types/page counts; declared damaged pages are
preserved in its report and are not silently changed. `config.json` is reserved
for later configuration; the current steps do not read it.

Manifest example:

```yaml
batch_id: 26_10_2026_Hoa
operator:
  name: Hoa
scan:
  date: 2026-10-26
  scanner_model: Scanner_Model
  resolution_dpi: 300
  color_mode: grayscale_8bit
  format: TIFF
  duplex: true
items:
  - box_code: B0012
    item_number: "01"
    item_type: folder
    total_pages: 120
    damaged_pages: []
  - box_code: B0012
    item_number: "02"
    item_type: book
    total_pages: 124
    printed_pages: 120
    damaged_pages: [15]
```

## Step 2: split into documents

Install the project's runtime dependencies once:

```powershell
python -m pip install -e .
```

Run Step 2 using the same batch root. It reads `<batch>\raw` and the Step 1
report, then writes to `<batch>\work\step-2-documents` by default:

```powershell
python -m ocr_pipeline split-documents `
  --batch "F:\ocr\batch01"
```

Use `--check-report` or `--output` to override those defaults.

For folder scans, the ZXing-C++ barcode reader treats a page with at least one
decodable barcode as a separator. Both sides of that duplex sheet are excluded,
even when only one side contains a readable barcode. Leading, trailing, and
adjacent separator sheets are reported as segmentation failures. Non-empty
ranges between separators become documents numbered `01`, `02`, and so on.
Book units are emitted as one document numbered `01` without barcode
detection. Units that failed Step 1 are recorded as `skipped`.

The output contains `split-report.json` and one `<box>-<unit>-documents.json`
file per unit. Each document record includes its ID, first and last scan page,
and explicit page list. The report also lists detected separator pages.

The separator barcode symbology is currently any format supported by
ZXing-C++; validate detection against the client's actual separator scans
before processing production batches.

## Step 3: flag blank pages

The pilot must determine the dark-pixel-share threshold. Pass the measured
threshold explicitly; no uncalibrated default is applied:

```powershell
python -m ocr_pipeline flag-blank-pages `
  --batch "F:\ocr\batch01" `
  --document-id "B0012-01-01" `
  --dark-pixel-share-threshold 0.01
```

The example value `0.01` is illustrative only and is **not** a recommended
production threshold. The command reads the Step 1 and Step 2 outputs by
default and writes one flag per page to
`<batch>\work\step-3-blank-pages\<document-id>.json`. Optional
`--split-report`, `--check-report`, and `--output` arguments override the
default paths.

The current pixel rule converts the first TIFF frame to 8-bit grayscale and
counts pixels with intensity below `128` as dark. A page is flagged blank when
its dark-pixel share is strictly less than the supplied threshold. The report
records both the share and threshold for auditability. Step 3 only emits flags;
downstream OCR must skip flagged pages while retaining their page markers.

## Step 4: OCR pages

Install the OCR dependencies from the project root:

```powershell
python -m pip install -e ".[layout,tesseract]"
python -m pip install --no-deps vietocr==0.3.13
python -m pip install gdown einops prefetch-generator lmdb
```

Run Step 4 after Steps 1-3:

```powershell
python -m ocr_pipeline ocr-document `
  --batch "F:\ocr\batch01" `
  --document-id "B0012-01-01" `
  --model vietocr `
  --language "vi+zh-Hant"
```

Use `vi` for Vietnamese-only pages or `vi+zh-Hant` for pages mixing Vietnamese
with Traditional Chinese. VietOCR recognizes Vietnamese line crops, Tesseract
detects text lines and recognizes Traditional Chinese, and PaddleOCR
PP-StructureV3 supplies layout regions only. PaddleOCR's recognized text is
discarded so it cannot override VietOCR's text. Figure, table, formula, stamp,
header, footer, and page-number regions are kept for downstream steps. Raw
responses from both OCR and layout passes remain local.

Step 4 logs each page's status and elapsed time as it completes. The same
`duration_seconds` value is stored per page in `ocr-report.json`, including
failed pages; blank pages record the time spent handling the skip.

VietOCR's `vgg_transformer.pth` weights are downloaded on first use into
`<project>\cache\vietocr`; PaddleOCR layout weights go into
`<project>\cache\paddlex`. Tesseract 5 and its `vie` and `chi_tra` language
data must be installed on the workstation. Set `TESSERACT_CMD` if
`tesseract.exe` is not on `PATH`. PaddleOCR is configured as a CPU layout
provider; MKL-DNN is disabled because this PaddlePaddle CPU runtime raised an
unsupported PIR attribute error with oneDNN. GPU PaddlePaddle is a separate
platform setup.

The adapter architecture remains extensible: other adapter distributions can
register an entry point in `ocr_pipeline.ocr_adapters` and implement the
`OCRModelAdapter` interface. The built-in `tesseract` model can also be chosen
for Tesseract-only text recognition; it does not run PaddleOCR layout, so Step
5 will not receive figure regions in that mode.

VietOCR's library is Apache 2.0; check the weights terms before production
use. On the page-37 sample, the mixed profile reads the Traditional Chinese
quotation `乾。元亨利貞。` and improves the title and Vietnamese diacritics
compared with Tesseract-only recognition. Validate the choice on a larger
sample with a reviewed reference before production.

### OCR quality tuning

The VietOCR and Tesseract adapters share a page preparation step before text
inference. The first TIFF frame is converted to grayscale, autocontrasted when
the page is low-contrast, upscaled when the page is small, and deskewed by up
to 5 degrees using a projection-profile search. The prepared image is never
written to disk and the original TIFF is never modified. Their text boxes are
normalized against the page frame. PaddleOCR layout inference keeps the
original scale and rotation so figure boxes map onto the master TIFF. Raw
outputs record the preprocessing metadata for both passes.

Engine-specific tuning:

- Tesseract uses the LSTM engine (`--oem 1`), automatic page segmentation
  (`--psm 3`), preserved interword spacing and a declared 300 DPI, and emits
  one region per visual line grouped into paragraphs by Tesseract's block and
  paragraph numbers.
- VietOCR uses its `vgg_transformer` model for Vietnamese line recognition;
  mixed mode sends Traditional Chinese lines to Tesseract `chi_tra`.
- PaddleOCR PP-StructureV3 provides layout and geometry only. Its text
  transcription is discarded so the final Markdown comes from VietOCR.

The page-37 trial favored the VietOCR/Tesseract combination for mixed
Vietnamese and Traditional Chinese. PaddleOCR remains in the pipeline only
because it supplies layout regions for downstream figure extraction. Confirm
the quality ranking on a larger sample with reviewed ground truth; Chandra
OCR 2 remains the separately specified production target.

## Step 5: cut out figures

After Step 4 succeeds, cut each `figure` region from its original TIFF page:

```powershell
python -m ocr_pipeline cut-out-images `
  --batch "F:\ocr\batch01" `
  --document-id "B0012-01-01"
```

By default, Step 5 reads Step 4 output from
`<batch>\work\step-4-ocr\<document-id>` and writes images under
`<batch>\work\step-5-images\<document-id>\images`. Optional `--ocr-output`,
`--output`, and `--jpeg-quality` arguments override the input/output locations
or JPEG quality.

Bounding boxes are normalized to the original page dimensions. Step 5 uses the
original TIFF for crops, keeps the full pixel resolution within each region,
and does not save bbox coordinates in the Step 5 report. Figure order follows
the Step 4 region order. Files are named `images/p<page>-<order>.<ext>`, for
example `images/p007-1.png`.

If Step 4 provides `image_kind: "drawing"`, output is PNG; if it provides
`image_kind: "photo"`, output is JPEG (default quality 95). If the model omits
this metadata or cannot distinguish the image kind, Step 5 chooses lossless PNG
rather than guessing. Cropping uses Pillow directly against the source TIFF at
its original pixel dimensions; no resize, perspective correction, or write to
the TIFF is performed. Invalid or out-of-page coordinates are reported as
failures, not clipped or guessed. Crops and their report do not retain region
bounding boxes. Blank pages do not produce figures. A failed Step 4 document
must be rerun before Step 5.

## Step 6: assemble the final document bundle

After Steps 1-5 and Step 7 succeed, assemble each document:

```powershell
python -m ocr_pipeline assemble-document `
  --batch "F:\ocr\batch01" `
  --document-id "B0012-01-01"
```

For an offline pipeline run that intentionally skips Step 7, add
`--skip-step7`. Step 6 then uses the figure label from Step 4 as alt text and
does not add generated descriptions or Mermaid diagrams:

```powershell
python -m ocr_pipeline assemble-document `
  --batch "F:\ocr\batch01" `
  --document-id "B0012-01-01" `
  --skip-step7
```

By default, Step 6 reads the Step 1, 2, 4, and 5 outputs and writes a portable
bundle under `<batch>\work\step-6-assemble\<document-id>`. The bundle contains
`document.md` and an `images\` directory. Optional `--split-report`,
`--check-report`, `--ocr-output`, `--image-output`, `--description-report`,
and `--output` arguments override those paths.

The Markdown contains YAML frontmatter, one scan-page marker for every page
(including blank pages), the OCR text, and links to extracted figures.
Adjacent plain-text paragraphs are conservatively joined across a page break
when the previous text has no sentence-ending punctuation and the next begins
with lowercase text; the next page marker remains before that page's remaining
content.
Frontmatter `title` uses an explicit OCR Markdown heading when available and
is otherwise empty; `type`/`language` are `unknown` until they have an
authoritative source. Repeated labeled page headers and footers are removed,
and recognized printed page numbers are included in scan-page markers.
Adjacent plain-text paragraphs are conservatively joined across page breaks.
Step 7 descriptions are required for every extracted figure unless
`--skip-step7` is explicitly selected; without Step 7, OCR figure labels
provide fallback alt text.
`pages` records the first and last scan page; `source` records the corresponding
first and last TIFF filenames. Failed OCR or image extraction/description
reports prevent assembly.

## Step 7: describe cropped images

The adapter accepts only PNG/JPEG crops from Step 5. It never reads or uploads
TIFF source pages. Install the optional Vertex client with
`python -m pip install -e .[vertex]`, configure Google Application Default
Credentials, then explicitly consent for a run:

```powershell
python -m ocr_pipeline describe-images `
  --batch "F:\ocr\batch01" `
  --document-id "B0012-01-01" `
  --project "your-project-id" `
  --location "us-central1" `
  --allow-cropped-image-upload
```

The default model is `gemini-2.5-flash`. Step 7 writes concise alt text and an
optional Mermaid flow/relationship diagram. The command refuses to contact
Vertex unless `--allow-cropped-image-upload` is provided. Offline tests use a
fake adapter and require no credentials or network access.

## Step 8: bundle and page quality

```powershell
python -m ocr_pipeline check-bundle `
  --batch "F:\ocr\batch01" `
  --document-id "B0012-01-01"
```

Step 8 checks UTF-8/NFC Markdown, valid YAML, all seven required metadata
fields, document ID, contiguous page markers, local image links, readable
image files, and non-empty alt text. Structural bundle status is independent
from page flags (`EMPTY_OCR`, `REPEATED_OCR`, `LOW_VOCABULARY`, `OCR_FAILURE`,
`TIMEOUT`). Flags are written to the JSON report and CSV log without
automatically rejecting a structurally valid bundle. A vocabulary ratio is
calculated only when both a vocabulary file and pilot-calibrated
`--vocabulary-threshold` are supplied.

## Step 9: local publication

```powershell
python -m ocr_pipeline publish-bundle `
  --batch "F:\ocr\batch01" `
  --document-id "B0012-01-01"
```

Only a bundle passing Step 8 is atomically promoted to
`<batch>\storage\output\valid`. Structurally failed bundles go to
`output\rejected` and never enter `valid`. Published bundles contain only
`document.md` and `images\`; intermediates and logs remain outside the bundle.
Existing destination IDs are never overwritten.

## Run provenance and generated end-to-end test

Every step accepts `--run-id`; reuse one value to group step records. Otherwise
a unique ID is generated. Input/output SHA-256 checksums, configuration version,
step status, and step configuration are stored at
`<batch>\work\runs\<run-id>.json`.
When a rerun changes its inputs, outputs, or configuration, dependent steps
already recorded under that run ID are marked `stale` in the same file and
must be rerun before publication.

Run a complete generated sample without an OCR server or cloud access:

```powershell
python -m ocr_pipeline self-test
```

This creates synthetic TIFFs (including a blank page and a figure), a manifest,
intermediate reports, a quality log, and a published bundle in a new
`ocr-pipeline-self-test-<run-id>` directory. Existing output folders are not
overwritten. The pipeline uses fake OCR and image-description adapters, so no
data leaves the workstation.

## Development

Run the test suite from the repository root:

```powershell
python -m unittest discover -s tests -v
```
