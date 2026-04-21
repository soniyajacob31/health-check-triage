# Health Check Triage

A patient-facing web app that guides people through plain-language questions about symptoms and health history, then suggests **where to seek care** (emergency department, urgent care, primary care, specialist, or self-care / watchful waiting). It is designed for **low medical literacy** and a **one-question-at-a-time** flow on mobile or desktop.

**Live demo:** [shouldigo.onrender.com](https://shouldigo.onrender.com)

---

## Purpose

Many people are unsure how urgently they need care. This app:

- **Collects** a short structured intake (demographics, free-text symptoms and conditions/meds, optional ZIP for nearby facilities).
- **Asks** a small number of follow-up questions tailored to the main complaint (from JSON “interview trees”).
- **Combines** a machine learning model trained on real emergency-department visit patterns with **hard-coded safety rules** and **conservative escalation** so borderline or high-risk patterns favor safer recommendations.
- **Explains** the suggestion with risk context, escalation “if this happens” guidance, a copyable summary for clinic staff, and (when appropriate) nearby ER/urgent care ideas via OpenStreetMap (browser-side; no API keys required).

**It is not a diagnosis and not a substitute for a clinician.** The UI includes disclaimers; use common sense and emergency services when appropriate.

---

## How it works (architecture)

| Layer | Role |
|--------|------|
| **Flask** (`app/routes.py`) | Serves pages: welcome → interview loop → results; manages session state; optional **Back** to undo the last answer. |
| **Interview engine** (`app/interview_engine.py`) | **`TreeInterviewEngine`** walks structured question trees in `app/config/interview_trees/*.json` (deterministic, no external LLM). A future **`LLMInterviewEngine`** stub could swap in for richer follow-ups. |
| **Patient state** (`app/patient_state.py`) | Accumulates answers, parses free-text symptoms/PMH into feature flags, and builds the feature vector for the model. |
| **Triage model** (`app/model.py`) | Loads trained classifiers (`app/models/*.joblib`), applies **red-flag overrides**, **probability-based escalation**, **PCP-first routing** for some complaints, and **specialist selection** using `app/config/complaint_specialist_map.json` (evidence-based mapping from ED literature). |
| **Evidence** (`app/evidence.py`) | Builds reassurance text, risk-style percentages (blending model output with published reference rates), escalation lines, triage-nurse summary, and related copy for `results.html`. |
| **Transcripts** (`app/database.py`) | Completed sessions can be saved to **SQLite** for research/review; a password-protected **admin** area lists and exports them (`/admin`). |

### Interview flow (simplified)

1. **Welcome** — Legal disclaimer and consent.
2. **Baseline** — Name, who the check is for (with immediate escalation if the helper indicates **confusion / altered mental status**), age, sex, free-text **symptoms** and **conditions/medications**, optional ZIP.
3. **Safety checks** — Configurable **red-flag** combinations (`app/config/red_flags.json`) after baseline (and after follow-ups where applicable) can short-circuit to an emergency recommendation.
4. **Follow-ups** — Up to **six** symptom-specific questions from the matching tree, or a **generic** tree for complaints without a dedicated file.
5. **Results** — Care level (five levels), evidence blocks, specialist/PCP referral wording as needed, facility hints if ZIP was provided.

Offline scripts **`build_triage_dataset.py`** and **`train_triage_model.py`** rebuild training data and models when you have the underlying data; the shipped app includes trained artifacts so you can run the site without retraining.

---

## Limitations

- **Trained on ED visit data** (e.g., MIMIC-style cohorts): populations and settings differ from every real-world user.
- **No vitals or labs** in the default patient flow—only what people can report—so accuracy is inherently limited compared to in-person triage.
- **Risk percentages** combine model output with **population-level** published statistics; they are illustrative, not personal medical forecasts.
- **Transcript storage** is optional in code terms but **enabled by default** when results load; treat deployments as handling **sensitive health-related free text** and secure the host, database, and admin routes accordingly.

For deeper technical detail (data pipeline, model training, safety layers, citations), see **`APP_DOCUMENTATION.txt`** in this repository.
