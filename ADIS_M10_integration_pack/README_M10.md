# ADIS M10 Integration Pack

## 1. Copy
Copy:
- `scripts/final_immune_loop.py` -> `<ADIS_ROOT>\scripts\final_immune_loop.py`
- `scripts/m10_final_audit.py` -> `<ADIS_ROOT>\scripts\m10_final_audit.py`
- `dashboard/app.py` -> `<ADIS_ROOT>\dashboard\app.py`

Do not replace the M3 v3 artifact or M4 policy.

## 2. Compile
```powershell
python -m py_compile scripts\final_immune_loop.py
python -m py_compile scripts\m10_final_audit.py
python -m py_compile dashboard\app.py
```

## 3. Run the final loop
```powershell
python -m scripts.final_immune_loop
```

Expected output ends with:
`[PASS] M10 END-TO-END LOOP EXECUTED`

## 4. Run audit
```powershell
python -m scripts.m10_final_audit
```

## 5. Run dashboard
```powershell
python -m streamlit run dashboard\app.py
```

Streamlit's documented module form is supported and opens the local app in a browser.

## Important
This pack deliberately keeps M3 v3 and M4 policy frozen. The production encoder remains:
`models/behavioral_encoder.joblib`

The loop uses the persistent `ImmuneMemoryAdapter`, so the memory path remains disk-backed rather than an in-memory demo store.
