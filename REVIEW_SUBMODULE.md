# Review: Submodule (costsci_tools) Cleanup

**Branch:** `doc/cleanup-runners-docs`
**Scope:** Fix broken runners, remove duplicate docs, fix typos

## Changes

### runners/compaction.py, runners/plate_with_a_hole.py
- **Added** `sys.path.append` block (4 lines each) — matches all other runners
- **Bug:** These two runners failed with `ModuleNotFoundError: No module named 'solvers'` when run from `costsci_tools/` directory, while every other runner worked fine
- **Verified** both now run successfully with default configs

### solvers/EPOCH_SETUP.md (deleted)
- Orphaned duplicate of `SimulCost-Bench/EPOCH_SETUP.md` (main repo)
- Nothing in the submodule linked to this copy
- Only diff was `cd /path/to/costsci-tools` vs `cd costsci_tools`

### solvers/EULER_2D_SETUP.md (deleted)
- Orphaned duplicate of `SimulCost-Bench/EULER_2D_SETUP.md` (main repo)
- Same situation — identical content, no internal references

### guideline_solver_checkout.md
- **Fixed** `tunnable` -> `tunable`
- **Fixed** broken sentence: "The checkout process uses delivers a config file and a section in the main document" -> "The checkout process delivers a config file and a documentation section"

## Verification
- `python runners/compaction.py verbose=true` -> prints cost + "Simulation completed"
- `python runners/plate_with_a_hole.py verbose=true` -> prints cost + "Simulation completed"
