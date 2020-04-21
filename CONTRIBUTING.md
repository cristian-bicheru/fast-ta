# Testing

All code added to the repository must pass unit tests (at a minimum). In order
to run our unit tests we build with bazel, although the final package will
be built using distutils from python.

To run all tests, call:
```bash
bazel test //fast_ta/src/...
```

If tests are failing, more specific info can be found by running:
```bash
bazel run //fast_ta/src:[TEST NAME]
```
this will write the gtest output to stdout.
