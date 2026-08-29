# Local trajectory manifests

Trajectory manifests are machine-specific because each JSONL record points to
local molecular-dynamics data. They are ignored by Git.

The canonical V4-D configs expect the mdCATH training manifest at:

```text
manifests/mdcath.train.jsonl
```

Validation and test manifests are passed explicitly to evaluation scripts so
that calibration and independent testing cannot be confused.
