# Edge/cloud v1 wire examples

**Design only. This contract is not connected to the runtime.** The schema is
[`../../contracts/edge-cloud-v1.schema.json`](../../contracts/edge-cloud-v1.schema.json)
and uses JSON Schema Draft 2020-12. All media, identifiers and hashes in these
examples are synthetic. No local files, private footage, GPS or upload endpoints
are included. Example artifact hashes do not identify real uploaded blobs.

Five message kinds share `protocol_version: 1`, `job_id`, `job_revision` and
`message_id`. Every wire object is closed with `additionalProperties: false`.

| Kind | Direction | Meaning |
|---|---|---|
| `manifest` | client → service | Registered asset metadata, slim advisory candidates, optional music, budgets and policy. |
| `evidence_request` | service → client | One typed `frames`, `video` or `evaluate_window` action within registered scope. |
| `evidence_receipt` | client → service | Generated artifacts and honest source/time/coverage provenance; a window evaluation provides bounded check summaries and an opaque local reference. |
| `edit_proposal` | service → client | Requested source windows, output frames, speed, crop and semantic intent. No supplied quality/profile/max values. |
| `validation_receipt` | client → service | A report about local validation, **not a render authorization token**. |

`manifest-estimate.json` is revision 0 before music selection (`music: null`,
`total_frames: null`). `manifest.json` is revision 1: a six-second synthetic song
at 48 kHz, 180 output frames at 30 fps, a four-second video followed by a
two-second photo. All other messages reference revision 1. The three request
examples demonstrate every action discriminator. The explicit single-time frame
request asks for 2s, and its ready receipt contains
exactly that one source frame (PTS 180000 at 1/90000). The evaluation receipt
references the evaluation request and includes client-produced check statuses for
planning. `ready` means the requested evidence is available, not that all checks
passed; inspect each check. For a pre-plan window evaluation `shot_id` is null,
and check names must match the requested set. These summaries are advisory when
echoed back in a proposal; only the local DB can resolve its own evaluation.
The video request intentionally has no fabricated receipt.

## Identity and trusted origin

IDs match a restricted opaque alphabet: no slashes, backslashes, dots, colon,
URL, shell or filesystem interpretation. Only the client registry maps an
`asset_id/source_revision` pair to a local content SHA and path. An unknown ID
must be rejected, never interpreted as a filename. A revision is not a
server-provided hash that the client should trust without its registry.

The `producer` string documents direction; **schema validation cannot prove who
sent a message**. Endpoints must enforce direction using the authenticated
connection. The edge accepts only service request/proposal kinds from the service.
It must not import a purported client receipt returned by the model as trusted
state. A model can copy a valid-looking ID, hash, `producer: client` or
`status: valid`; none establishes authenticity.

`local_evaluation_ref` and `local_receipt_id` refer to existing trusted client DB
rows, bound to the current job/revision, source revisions, exact scope and
compiled plan. Uploading or echoing these references grants no execution rights.
The client recompiles the proposal, computes its own assessments and checks its
own current DB before any local operation. Actual render authorization remains
local; this wire schema defines no arbitrary render command or remote path.
The validation example is deliberately `needs_review`: video technical checks
are recorded, but the photo has no composition evidence, its check is `unknown`,
and it has a `MISSING_EVIDENCE` violation. `local_receipt_id` and
`compiled_plan_sha256` remain null. It must not be treated as completed validation
or renderable output. All recorded technical values are synthetic illustrations,
not claims that real media was tested.

## Time, scope and numeric rules

- Seconds use `{num, den}` integers. `num >= 0`, `den >= 1`; fractions must be
  reduced by gcd, including zero as `0/1`. Positive speed, FPS fractions and
  native time bases require `num >= 1`. JSON Schema enforces integer bounds and
  zero normalization; it cannot enforce general gcd reduction or fraction order.
- Native `actual_pts`/`presentation_start_pts` may be signed integers; their time
  base is positive. Source windows are nonnegative time relative to the displayed
  asset origin. The local mapper accounts for the native PTS origin. Source FPS
  is rational, output FPS is an integer from 24 to 60, and output intervals are
  integer frame intervals.
- All windows and output frame ranges are half open `[start,end)` and require
  `end > start`. Rational comparisons must be exact, not rounded display seconds.
  Native PTS and output frame IDs are different coordinates. Never invent an
  original frame ID from a rendered-output frame.
- Each evaluation scope binds asset/revision, exact consumed window, speed,
  output FPS, transform, ruleset, analyzer, measurement and timing revisions.
  Do not inherit a parent window or a different output FPS as exact support.
  A changed prefix can change a local trend calculation even inside a good parent.
- Source consumption must equal output-frame duration × speed. A source reservation
  may have extra margin, but its broader evaluation cannot authorize the actual
  consumed subwindow. Validate actual supported frames and do not freeze a frame
  or extend a source silently to fill the music.
- Video transforms are forward-only in v1. Photo scope/proposal windows are null
  and photo speed is exactly `1/1`, enforced by the schema. Asset type matching,
  absence of video timing on a photo, and local photo-specific evaluation still
  require cross-message checks. Photos have no video playback profile.
- Crop coordinates use the EXIF/display-oriented image in 1/10000 units. Local
  validation checks `x+width <= 10000`, `y+height <= 10000`, output fit and supported
  zoom bounds. A geometrically valid crop can still cut a meaningful subject;
  start/end composition previews and evidence-backed review are separate.
- `verified_cfr` requires concrete FPS, native time base, frame count, PTS origin
  and verified index mapping. `vfr`/`unknown` can appear in a manifest; that does
  not mean the current analyzer can assess them. Unsupported timing must return
  insufficient/unsupported rather than fabricate a CFR assessment.
- Channels preserve measured/derived/unavailable/unknown provenance. SRT may
  populate zero-valued gimbal arrays without measuring gimbal angles; field
  presence or zeros are not proof of sensor availability.

## Cross-message checks outside JSON Schema

A validator must check registered asset/candidate/music IDs and revisions;
request/receipt correlation; job revision freshness; stable-range containment;
exact evaluation scope; time-base mapping; source capacity; candidate-to-shot
asset agreement; evidence-to-scope agreement; chronological order; adjacent
rotation; full contiguous output frame coverage; music samples/duration/beat-slot
alignment; photo/video transform compatibility; and all local policy decisions.

For this example, video source `[1,5)` consumes four seconds at 1x for output
`[0,120)`, then the photo occupies `[120,180)`. The six-second song has 288,000
samples. Digest canonicalization is **JCS, RFC 8785**, with these exact hash domains:

```text
scope_sha256 = SHA256(ASCII("dronesift.scope.v1") || 0x00 || JCS(scope))
proposal_sha256 = SHA256(ASCII("dronesift.proposal_message.v1") || 0x00 || JCS(complete immutable edit_proposal message))
compiled_plan_sha256 = SHA256(ASCII("dronesift.compiled_plan.v1") || 0x00 || JCS(local semantic compiled plan))
```

The proposal digest includes its full immutable transport message, including
job/revision/message headers. The compiled-plan digest excludes transport headers,
request/message IDs, upload state, timestamps, local paths, preview refs and prose.
It binds the local normalized semantic plan: asset/source revisions, exact source
consumption, frame timeline, speed/transforms, output, music identity/sample count,
and rule/analyzer revisions. The local receipt separately binds job identity and
revision. No compiled-plan hash is emitted for this unresolved example.

Reject noncanonical fractions before hashing; JSON does not permit NaN/Infinity.
The current all-ASCII-key/string and integer-only hashed example values happen to
produce the same JCS bytes with sorted keys and compact UTF-8 JSON separators. That
is **not** a general JCS implementation (Unicode sorting and numeric serialization
need a conforming implementation). Hashes are computed locally; a digest or valid
schema cannot prove that a measurement/rendered file exists.

A `validation_receipt` with `status: valid` is structurally constrained to a
compiled-plan hash, no violations, and pass/not-applicable check statuses. It is
still only a report. Current runtime behavior is unchanged, including legacy
natural-speed advisory compatibility; a remote proposal importer needs its own
strict local preflight.

## Evidence and budgets

Evidence contains an object ID, hash, byte count, origin and coverage; there is
no arbitrary upload/download URL. Blob transfer must use authenticated,
project-scoped storage configured by the client/service. Sparse frames or cached
sprites never establish complete stability certification. Cached sprite times
must remain approximate unless actual source PTS are known. The exact-source
sample variant requires actual PTS and time base. AI visual observations are not
local-user continuous playback reviews.

Manifest `max_frames_per_request` and `max_total_image_pixels_per_request`
apply to still-image requests only. Frames have fixed size presets:
`max_total_image_pixels` sums the delivered still
images in that request, and `max_frames` bounds their count. A video uses
`max_pixels_per_frame`, **not** a sum over all decoded frames; its primary limits
are dimensions, `max_output_duration`, bytes and the cumulative proxy-seconds
ledger. Four seconds at 30fps/640x360 is 27,648,000 decoded pixel instances but
230,400 pixels per frame, so it fits the example's 518,400 per-frame limit. Proxies
preserve the planned output FPS and have no audio. Requests must be checked against
the manifest's matching image/proxy/duration/byte limits. Metadata and binary
evidence payload byte ceilings are distinct. The schema bounds each payload's
shape and some numeric limits; **it does not enforce cumulative resource usage**.
Client and service need a shared idempotent budget ledger for total upload bytes,
proxy seconds, pixels, model context/cost and retries. Reject oversized/decompression
bomb inputs before decoding. Repeating requests or changing IDs must not bypass
job totals. Missing/unsupported evidence remains unknown; never pad values with
zeros or emit a fabricated successful assessment.

The original media and original MP3 stay local. Music metadata is sufficient for
beat-aligned planning; `music: null` is valid for inventory and duration estimates.
The service may propose edits but cannot use text/OCR/filenames as tool commands.

## Validation

Use a temporary validator environment without changing `pyproject.toml` or
`uv.lock`:

```sh
uv run --no-project --with 'jsonschema>=4,<5' python - <<'PY'
import json
from pathlib import Path
from jsonschema import Draft202012Validator
schema = json.loads(Path("docs/contracts/edge-cloud-v1.schema.json").read_text())
Draft202012Validator.check_schema(schema)
validator = Draft202012Validator(schema)
examples = sorted(Path("docs/examples/edge-cloud-v1").glob("*.json"))
for path in examples:
    validator.validate(json.loads(path.read_text()))
print(f"Validated schema and {len(examples)} synthetic positive examples")
PY
```

This command was run successfully during preparation. JSON syntax, local `$ref`
resolution, wire-object closure, example references and the example hash domains
were also checked. These are structural checks, not proof of media quality,
completed review, trustworthy sender identity or local authorization. Schema
validation does not replace the cross-field/client DB checks above.
