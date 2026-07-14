# Dataset Manifests

## CWRU Bearing Data

`cwru_manifest.json` describes the 30 recordings used for the ten-class CWRU
bearing task: ten health/fault classes at motor loads 1, 2, and 3 hp. The file
conditions and source URLs come from the Case Western Reserve University
Bearing Data Center.

The adapter automatically downloads missing manifest files into
`cache/cwru/raw/` when `config.dataset.download` is enabled. Existing files and
processed caches are reused. The adapter supports drive-end (`DE`) and fan-end
(`FE`) signals independently. It downsamples 48 kHz normal drive-end signals to
12 kHz and leaves 12 kHz signals unchanged.

The window protocol is a reconstruction of RES-HD because the publication does
not report its exact stride and train/test split procedure. For every
class/load recording, the adapter reserves a tail region for deterministic test
windows and samples training windows without replacement only from the region
before it. The default configuration produces 19,800 training windows and 750
test windows of 100 samples each.
