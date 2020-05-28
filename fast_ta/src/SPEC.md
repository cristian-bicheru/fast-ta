# Coding Specifications
## General Specs:
 * A quick docstring should be provided for each function in its respective header file.
 * Functions should generally try to return/store to aligned arrays. This because we handle the creation of return arrays in our code, and they are always aligned. When returning 2D arrays this is not possible, as NumPy arrays are memory contigous by spec. But, in all other cases, aligned store instructions should be used.
 * Non-temporal store intrinsics are used whenever possible due to the nature of the algorithms and the speedup they provide.

## Functions Ending In `_unaligned`:
 * These functions do not expect aligned input arrays, but do expect the output array to be aligned (at least by the size of the data type).

## Functions Not Ending In `_unaligned`:
 * These functions expect all arrays to be aligned (at least by the size of the data type).

## `STATIC_ASSERT` Usage:
 * `STATIC_ASSERT` takes two arguments, the first of which is a standard, compile-time evaluable expression (e.g `sizeof(double)==2*sizeof(float)`) and the second of which is a message to output on assertion failure.
 * The message must not contain any spaces or special characters.
 * On assertion failure, the compiler will output: `error: conflicting types for ‘STATIC_ASSERTION_FAILURE_<MESSAGE>_AT_LINE_<LINE NUMBER>`.
