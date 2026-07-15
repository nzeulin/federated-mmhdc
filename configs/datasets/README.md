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

Each selected recording is split in timestamp order after resampling. By
default, the first 70% of timestamps form the training region and the remaining
30% form the test region. Complete 100-sample windows are enumerated separately
within each region using configurable train and test strides, so no window
crosses the split boundary. Incomplete trailing windows are discarded.
