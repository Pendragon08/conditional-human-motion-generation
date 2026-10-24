# Deterministic execution checkpoint

Both frozen-r3 document commands completed once, without repair or rerun. The frozen document matched SHA256 `5e1ccc5b0ae9898bb2d32dfdff7116abb77a44f4674ee5faa63cf5af47e3b34a`. Remote/local new-code hashes matched: semantics `6c073d04ce2dcb1fa8cf4ad3076a2c526d0453c37bf5608cb11ceab3107146e9`; batch loader `fb9c3b8032abb7206c825743bf1204f7993489e6f947bba7b806ce7329db871f`. Before execution, semantic output, derived legacy-core output, and batch-loading output were all absent.

The semantic command exited 0 with 3,853-byte stdout SHA256 `85d0b182134ce0dd21b67142c9a03b12b496dda7656d33c1c26362af8142cc34` and empty stderr. Its receipt was 25,495 bytes, SHA256 `cb924d86939df1d8738aca0c1ff7142b1d130664a2db598f764beb60f1684d23`; the derived legacy receipt was 883 bytes, SHA256 `b60eca940139cd79fe878612f47346b36c12a9a588d21a0a637dee150eb3109c`.

The batch-loading command exited 0 with 777-byte stdout SHA256 `86aedb0a407d3bd9c83e539dfe0cb2c7fa124f67e9b33201722beaa42e12a619` and empty stderr. Its receipt was 819 bytes, SHA256 `7c1ef57a0ad4c98d45f1464639fa3750b034fb4fc4aa9714f995e20a2de6d59a`.

No documented-command divergence occurred. The remote clock was approximately 1.8 seconds ahead of the client clock.
