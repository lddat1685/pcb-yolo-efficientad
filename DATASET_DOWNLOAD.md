# Dataset archive

`DatasetVer4_Public.zip` contains the dataset used by this project, including
images, labels and the canonical split manifests (2,998 DeepPCB samples):

| Partition | Images |
| --- | ---: |
| train | 1,792 |
| calibration (validation) | 460 |
| fusion | 306 |
| test | 440 |

The ZIP is stored with Git LFS. After cloning, install Git LFS and run:

```sh
git lfs install --local
git lfs pull
```

Alternatively, open `DatasetVer4_Public.zip` on GitHub and use the download button.
Verify its SHA-256 against `DatasetVer4_Public.zip.sha256`.
Extract it into the parent directory of this repository so the layout is:

```text
parent/
  pcb-yolo-efficientad/
  DatasetVer4_Public/
    benchmarks/deeppcb/
```

The default dataset configuration detects this location automatically.
For a different location, set `DATASET_ROOT` to the extracted
`DatasetVer4_Public` directory. Keep the canonical split manifests unchanged.

The archive may contain additional source images and alternate dataset views;
the table above describes the canonical benchmark manifest used by this project.
