# Postoperative Spinal Infection Risk Calculator

Public clinician-facing Streamlit app.

## Deploy on Streamlit Community Cloud

1. Push this folder to a GitHub repository.
2. Open https://share.streamlit.io/deploy.
3. Select the GitHub repository and branch.
4. Set the main file path to:

```text
spinal_infection_public_app_v3/app.py
```

5. Deploy. Streamlit will install the packages from:

```text
spinal_infection_public_app_v3/requirements.txt
```

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The public page does not show model parameters, resampling details, or technical thresholds. It presents the result as an estimated probability with cohort-average context and clinical interpretation.
