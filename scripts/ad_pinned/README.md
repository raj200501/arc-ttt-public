# LiveBench scorer code, vendored for Addendum AD

These files are copied byte for byte from
[LiveBench/LiveBench](https://github.com/LiveBench/LiveBench) with
`git show <commit>:<path>`. They are not edited, formatted or linted.
Addendum AD (`docs/research/ADDENDUM_AD_PROTOCOL.md`) executes them as
shipped, through `scripts/published_number_audit.py`, which loads each
file by path and never imports this directory as a package.

Licence: LiveBench's `LICENSE` is copied beside them unchanged
(sha256 `8fb1d09632b9c03b426c87654d62c5d8fb953643e51e6866f3446401903005c3`). It is the Apache License 2.0 as copied from
lm-sys/FastChat, unchanged since LiveBench's first commit. LiveBench has
no NOTICE file. No file here is modified, so no change notice applies.

Every digest below is also in `experiments/ad_manifest_2026-10-21.json`
(`pinned_files`, `pinned_sources`), and the protocol pins each file by
its sha256. The `fetch` stage downloads each file again from
`raw.githubusercontent.com` at its commit and refuses unless the bytes
match.

| File | Source | Versions that use it | sha256 |
|---|---|---|---|
| `gen_ground_truth_judgment_19c7757.py` | `livebench/gen_ground_truth_judgment.py` @ `19c7757` | 19c7757, c5c293c | `c414031603e32ae3868299149b90241bc5841501adf38b38414ffaecbd08b6db` |
| `gen_ground_truth_judgment_905b71f.py` | `livebench/gen_ground_truth_judgment.py` @ `905b71f` | 905b71f | `b35746a0003db176914a5e3fa33496276c66ebc666156a5a57e2284c93dfa220` |
| `gen_ground_truth_judgment_e647d72.py` | `livebench/gen_ground_truth_judgment.py` @ `e647d72` | e647d72 | `a6882e2f3bbc7679601aa776b4dec8318d233935d4b3b7e4d8e3b408f2a4b746` |
| `gen_ground_truth_judgment_f3501c7.py` | `livebench/gen_ground_truth_judgment.py` @ `f3501c7` | f3501c7 | `5d0b7a28d3b8f042d2eae7ee192e92d8c56c949c635a181441fb8f01ab22ccac` |
| `gen_ground_truth_judgment_f774d19.py` | `livebench/gen_ground_truth_judgment.py` @ `f774d19` | f774d19 | `81226f3dfc4c06b3c4a168000dd55b1e868a992565ed365d3a6c37bd81d1e3ba` |
| `gen_ground_truth_judgment_fa6a3c6.py` | `livebench/gen_ground_truth_judgment.py` @ `fa6a3c6` | fa6a3c6 | `ed7973d19360ccf3a4a58d3112918b4f561009cdbdd2c4df55a2c85525a3d137` |
| `tablejoin_utils_19c7757.py` | `livebench/process_results/data_analysis/tablejoin/utils.py` @ `19c7757` | 19c7757 | `b04aee527f1e38d8a2dad456d2bcf2b2af622d64cd01221a3f31a480bb54c0e8` |
| `tablejoin_utils_905b71f.py` | `livebench/process_results/data_analysis/tablejoin/utils.py` @ `905b71f` | 905b71f | `ee638884f08ea802f59d14583f77c7e38b0c045e8d6bf309af0cca7d7bf12f3a` |
| `tablejoin_utils_c5c293c.py` | `livebench/process_results/data_analysis/tablejoin/utils.py` @ `c5c293c` | c5c293c, fa6a3c6, f3501c7 | `47a6fe5b2277b5760a186e9c556023e90a291bf915715dc5b87d0dba9e8bc0eb` |
| `tablejoin_utils_e647d72.py` | `livebench/process_results/data_analysis/tablejoin/utils.py` @ `e647d72` | e647d72 | `1d8e56395b6ad033391503fe5914eb242f080eed5f07695c655e28d779dd213c` |
| `tablejoin_utils_f774d19.py` | `livebench/process_results/data_analysis/tablejoin/utils.py` @ `f774d19` | f774d19 | `8d7fa703061d4536ee9f07c1a3d103690a6a20c01d8ab2c284fcb43ce45724b9` |
| `util_19c7757.py` | `livebench/process_results/util.py` @ `19c7757` | 19c7757, c5c293c, fa6a3c6 | `e8f97597589c640a89e1358093f9aa5d90f0673fedbc3998fcc267ce0854c0f1` |
| `util_f3501c7.py` | `livebench/process_results/util.py` @ `f3501c7` | f3501c7, e647d72 | `f4f8769b5b0d1bf696151212962823009753b822e3310c2087818f7a36b1ac0a` |

`gen_ground_truth_judgment_*.py` is never run as a module. The runner
lifts, by syntax tree, the consecutive assignments to `llm_answer` inside
`play_a_match_gt` (the answer extraction), and runs only those
statements. `tablejoin_utils_*.py` and `util_*.py` are run as modules.
