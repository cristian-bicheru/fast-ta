# Changelog
A summary of notable changes to the Fast TA library will be documented in this file.

# Pre-Release
## v0.1.4 - May 27, 2020
### Changed
 * Converted eligible operations to aligned operations.
 * Changed the functionality of the RSI algorithm in the first window to match that of the TA library.
 * Reorganized fast_ta/src folder.
 * Increased memory efficiency of several algorithms.
 * Added testing tolerance for double precision.
### Added
 * Added changleogs/CHANGELOG.md (changelog).
 * Added fast_ta/src/SPEC.md (coding specifications).
 * Added `core.instruction_set` function to show the SIMD instruction set Fast TA was compiled with.
 * Added `core.align` function to align a numpy array to the required byte boundary.
 * Added fast/ta/src/funcs_unaligned{.h/.c} (see SPEC.md).
### Removed
No Removals.
