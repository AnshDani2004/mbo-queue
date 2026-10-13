`databento_sample_GLBX_mbo.dbn.zst` is not stored in this repository. `scripts/fetch_test_data.sh`
downloads it from the public test suite of databento-python (Apache-2.0), pinned to commit
`130cb8a576a7a039f5e14528f13d2da09dcf0725`, and checks its SHA-256:
https://github.com/databento/databento-python/blob/130cb8a576a7a039f5e14528f13d2da09dcf0725/tests/data/GLBX.MDP3/test_data.mbo.dbn.zst

It holds 4 MBO records and is used only to test the real DBN decode path. The test that reads it
is skipped when the file is absent.
