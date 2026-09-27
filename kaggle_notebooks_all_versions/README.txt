Kaggle notebooks, all versions (packed 25 Sep 2026, 22:55 IST)

For every notebook: Create -> New Notebook -> File -> Import Notebook -> Add Input "er-challenge-data"
-> set Accelerator + Internet as below -> Save Version -> Save & Run All (Commit).
In the log, look for "BEST val F0.5" (v1 = 0.9529). Good runs end with "PASS". Download output/matching_results.tsv.

File                                 Accelerator     Internet  Our val F0.5        Expected Kaggle time
v1_entity_resolution_kaggle.ipynb    None (CPU)      On        0.9529 (public 0.9405)  ~3 h
entity_resolution_v2_kaggle.ipynb    None (CPU)      On        0.9568              ~4-5 h   (a stronger v2 is coming)
entity_resolution_v3_kaggle.ipynb    GPU T4 x2       On        from Kaggle log     ~5-6 h
entity_resolution_v4_kaggle.ipynb    GPU T4 x2 (not P100)  On  from Kaggle log     ~5-7 h
entity_resolution_v6_kaggle.ipynb    None (CPU)      On        0.9618              ~7 h
entity_resolution_v7_kaggle.ipynb    None (CPU)      On        0.9647              ~4-5 h
entity_resolution_v8_kaggle.ipynb    None (CPU)      On        0.9673 (best single model; blend pending)  ~6 h

v5 is not ready yet (still training).
