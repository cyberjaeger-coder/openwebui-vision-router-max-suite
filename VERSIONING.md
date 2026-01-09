# Versioning Guidelines

## Canonical version source
- `VERSION` is the single source of truth for the suite version.

## Where to update
- When the suite version changes, update:
  - `VERSION`
  - The `version:` header in:
    - `filters/openwebui-vision-router-max-filter.py`
    - `pipes/vision_followup_pipe.py`
    - `pipes/graph_followup_pipe.py`

## What to change for each release
- **Patch** bumps (x.y.Z): bug fixes, performance improvements, or internal refactors.
- **Minor** bumps (x.Y.z): new features or new configuration valves.
- **Major** bumps (X.y.z): breaking changes or incompatible behavior updates.
