# Implementation Plan: Cancer Mutation Pathogenicity Predictor — Feature Expansion & UI Overhaul

## Context

The current Streamlit dashboard has a heavy glassmorphism dark theme that looks cluttered, an unnecessary API Docs page, and lacks clinical utility features. The user wants to transform this from a pure research demo into a more patient/doctor-facing tool with treatment guidance, doctor booking, and patient profiles — while keeping the research ML core intact.

**Current state:** 7 Streamlit pages, FastAPI backend, PyTorch model, no auth/profiles/booking.
**Target state:** Clean minimalist UI, 9 pages (remove API Docs, add Cure Options + Book Doctor + Patient Profiles), integrated treatment knowledge base.

---

## Phase 1: Minimalistic UI/UX Redesign

**Why first:** Every subsequent phase creates new pages/components — they should use the new design from the start.

**Design direction:** Light, clean, clinical. White cards with subtle borders, indigo accent (`#4F46E5`), Inter font with dark text. No glassmorphism, no blur, no neon glow.

### Files to modify

**webapp/utils/styling.py** — Complete rewrite of `get_custom_css()`:
- Add a `THEME` dict at top with all color constants (bg, card, text, accent, etc.)
- Replace `.stApp` background: `#0F172A` gradient → `#FAFAFA` solid
- Replace `.glass-card`: `rgba(30,27,75,0.5)` + `backdrop-filter:blur` → `white` + `1px solid #E2E8F0` + light shadow
- Replace `.dashboard-header`: gradient purple → flat white card with indigo left-border accent
- Replace `styled_metric_card()`: dark backgrounds → white cards with accent `border-top`
- Update all text colors: `#F1F5F9` → `#1E293B` (headings), `#94A3B8` → `#64748B` (muted)
- Remove all `backdrop-filter: blur()` declarations
- Simplify buttons: solid `#4F46E5`, white text, no glow
- Update input fields: white bg, `#E2E8F0` borders

**webapp/app.py** — Sidebar brand & status:
- Lines 51-73: Update sidebar logo from dark-theme colors to dark-on-light
- Lines 86-116: Update API status badges for light background

**All 6 view files** — Update inline HTML styles:
- webapp/views/home.py — header, metric cards, quick start guide
- webapp/views/single_prediction.py — prediction cards, expanders, badges
- webapp/views/batch_analysis.py — results table, summary cards
- webapp/views/model_performance.py — metrics cards, chart theme
- webapp/views/data_explorer.py — overview cards, gene browser
- webapp/views/about.py — overview cards, link cards

**Pattern:** In each view, find all `color:#F1F5F9`, `color:#94A3B8`, `color:#A5B4FC`, `background:rgba(30,27,75,...)` and update to light-theme equivalents. Update `PLOTLY_DARK` constants to use dark text on transparent bg.

---

## Phase 2: Remove API Docs Page

**webapp/app.py**:
- Remove `api_docs` from import block (line 14)
- Remove `"API Docs": api_docs` from `PAGES` dict (line 46)

**webapp/views/api_docs.py**: Delete this file (500 lines)

**webapp/views/about.py**: Keep the Swagger link card but update text to say "API docs available at FastAPI /docs endpoint"

---

## Phase 3: Cancer Precautions & Cure Options  ✅ COMPLETED

### New files

**`webapp/data/__init__.py`** — empty package init

**`webapp/data/cancer_knowledge.py`** (~350 lines) — Structured knowledge base:
```python
CANCER_KNOWLEDGE = {
    "Breast Invasive Carcinoma": {
        "abbreviation": "BRCA",
        "overview": "...",
        "precautions": [
            {"category": "Screening", "detail": "Regular mammography..."},
            {"category": "Lifestyle", "detail": "..."},
            {"category": "Genetic", "detail": "BRCA1/BRCA2 testing..."},
        ],
        "treatment_options": [
            {"name": "Surgery", "description": "...", "stage": "All stages"},
            {"name": "Chemotherapy", "description": "...", "stage": "Stage II+"},
            ...
        ],
        "survival_rates": {"Stage I": "99%", ...},
        "key_genes": ["BRCA1", "BRCA2", "ERBB2", "TP53", "PIK3CA"],
        "clinical_trials_url": "https://clinicaltrials.gov/search?cond=...",
    },
    # Same structure for: Lung Adenocarcinoma, Colorectal, Uterine, Ovarian
}

PATHOGENICITY_GUIDANCE = {
    "Pathogenic": {
        "severity": "High",
        "recommendation": "Consider genetic counseling and specialist referral",
        "follow_up": ["Confirmatory testing", "Family screening", "Specialist consultation"],
    },
    # Same for: Likely Pathogenic, Benign, Likely Benign
}
```

**`webapp/views/cure_options.py`** (~250 lines) — New page:
- Cancer type dropdown selector (5 TCGA types)
- Overview section
- Precautions displayed as category cards (Screening, Lifestyle, Genetic)
- Treatment options as expandable cards with stage-applicability badges
- Survival rate visualization (bar chart or table)
- Links to clinical trials & NCCN guidelines
- Key associated genes (clickable links to Data Explorer)
- Disclaimer banner: "Research tool — consult healthcare professionals"

### Integration points

**webapp/views/single_prediction.py** — After prediction results:
- Add "View Treatment & Precautions" expander when cancer_type is known or prediction is Pathogenic/Likely Pathogenic
- Display relevant `CANCER_KNOWLEDGE` inline + link to full Cure Options page

**webapp/views/batch_analysis.py** — After results summary:
- For pathogenic results with cancer type, show link to Cure Options page

### Optional API endpoint

**`api/routes/knowledge.py`** (~60 lines):
- `GET /knowledge/cancer-types` — list cancer types with summaries
- `GET /knowledge/cancer-types/{type}` — full precaution/treatment data

**api/main.py** — Include the new knowledge router

---

## Phase 4: Patient Profile System

### Storage

**`webapp/data/patient_store.py`** (~250 lines) — SQLite-based storage (`data/patients.db`, gitignored):

Tables:
- `patients` — id, name, email, phone, dob, gender, blood_group, medical_history (JSON), cancer_type, timestamps
- `patient_reports` — id, patient_id, report_type, report_data (JSON), file_name, timestamp
- `patient_predictions` — id, patient_id, variant_id, predicted_class, confidence, cancer_type, request/response JSON, timestamp
- `appointments` — (used by Phase 5)

Class `PatientStore` with CRUD methods: create/get/update/delete patient, save/get predictions, save/get reports.

### New page

**`webapp/views/patient_profiles.py`** (~350 lines):
- **Tab 1 — Create Profile**: Form with name, email, phone, DOB, gender, blood group, medical history (text area), cancer type selector
- **Tab 2 — Search/Browse**: Search box, patient cards with key info, click to view details
- **Tab 3 — Patient Detail** (session-state driven): Full info card, prediction history table, uploaded reports, "Run New Prediction" button

### Integration

**webapp/app.py**: Initialize `PatientStore` in `st.session_state`

**webapp/views/single_prediction.py**:
- Add collapsible "Link to Patient" section with patient dropdown
- Auto-save prediction to `patient_predictions` when patient is linked
- "Save to Patient Profile" export button

**webapp/views/batch_analysis.py**:
- "Save Results to Profile" button after batch completes

---

## Phase 5: Book a Call with Doctors

### Mock doctor data

**`webapp/data/doctor_database.py`** (~200 lines):
- List of 15-20 mock doctors with: name, specialty, cancer_types covered, qualifications, experience, hospital, available_slots (day + times), consultation_fee, rating, bio
- Doctors cover all 5 cancer types (3-4 per type)
- Helper functions: `get_doctors_by_cancer_type()`, `get_doctor_by_id()`, `get_available_slots()`

### Booking storage

**`webapp/data/patient_store.py`** — Add `appointments` table:
- Fields: id, patient_id (optional), patient_name, email, phone, doctor_id, cancer_type, date, time, reason, status, notes, timestamp
- Methods: `create_appointment()`, `get_appointments_by_patient()`, `cancel_appointment()`

### New page

**`webapp/views/book_doctor.py`** (~300 lines):
- **Filter bar**: Cancer type dropdown + specialty filter
- **Doctor cards** (2-column grid): Name, specialty, qualifications, experience, rating stars, hospital, availability badges, "Book" button
- **Booking form** (appears on doctor select): Patient name/email/phone (pre-filled if profile linked), date picker, time slot selector, reason textarea, "Confirm" button
- **Confirmation panel**: Booking details + appointment ID
- **My Appointments tab**: History of bookings (by session/email)

### Integration

**webapp/views/single_prediction.py**: After Pathogenic/Likely Pathogenic results, show "Consult a Specialist" button → navigates to Book Doctor with cancer type pre-selected

**webapp/views/cure_options.py**: "Book a Consultation" CTA at the bottom of each cancer type section

---

## Phase 6: Additional Features

### 6a. Risk Assessment Summary
Add to webapp/views/single_prediction.py:
- Visual risk score card combining prediction confidence, uncertainty, gene driver status, mutation severity
- Color-coded risk level (Low / Moderate / High / Critical)
- Actionable next steps based on risk level

### 6b. Prediction History (Session-based)
Add `st.session_state["prediction_history"]` in single prediction view:
- Track all predictions made during the session
- "Compare with Previous" toggle showing side-by-side comparison table

### 6c. Enhanced PDF Report
**webapp/utils/report_generator.py**:
- Add treatment/precaution section when cancer type is known
- Add recommended specialist doctors section
- Add prominent research disclaimer

### 6d. Research Disclaimer
Add persistent footer banner across all pages:
- "This tool is for research purposes only. Not for clinical decision-making."
- Implement in `get_custom_css()` as a fixed-position footer or via `app.py`

---

## Final Navigation Structure

```python
PAGES = {
    "Home": home,
    "Patient Profiles": patient_profiles,     # NEW
    "Single Prediction": single_prediction,
    "Batch Analysis": batch_analysis,
    "Cure Options": cure_options,              # NEW
    "Book a Doctor": book_doctor,             # NEW
    "Model Performance": model_performance,
    "Data Explorer": data_explorer,
    "About": about,
    # API Docs — REMOVED
}
```

Sidebar grouping with section dividers:
- **Patient Care**: Patient Profiles, Single Prediction, Batch Analysis
- **Clinical Info**: Cure Options, Book a Doctor
- **Research**: Model Performance, Data Explorer, About

---

## New Files Summary

| File | Purpose | ~Lines |
|------|---------|--------|
| `webapp/data/__init__.py` | Package init | 1 |
| `webapp/data/cancer_knowledge.py` | Cancer treatment/precaution KB | 350 |
| `webapp/data/doctor_database.py` | Mock doctor database | 200 |
| `webapp/data/patient_store.py` | SQLite patient/appointment store | 300 |
| `webapp/views/cure_options.py` | Cure Options page | 250 |
| `webapp/views/book_doctor.py` | Book Doctor page | 300 |
| `webapp/views/patient_profiles.py` | Patient Profiles page | 350 |
| `api/routes/knowledge.py` | Knowledge API endpoints | 60 |

## Modified Files Summary

| File | Changes |
|------|---------|
| `webapp/utils/styling.py` | Complete CSS rewrite (light theme) |
| `webapp/app.py` | New imports, PAGES dict, sidebar grouping, patient store init |
| `webapp/views/home.py` | Theme colors, quick links to new features |
| `webapp/views/single_prediction.py` | Theme + cure/doctor/patient integrations |
| `webapp/views/batch_analysis.py` | Theme + patient linking + cure links |
| `webapp/views/model_performance.py` | Theme colors + chart theme |
| `webapp/views/data_explorer.py` | Theme colors |
| `webapp/views/about.py` | Theme + update API docs reference |
| `webapp/utils/report_generator.py` | Add treatment section to PDF |
| `api/main.py` | Include knowledge router |
| `api/schemas.py` | Add knowledge Pydantic models |

## Deleted Files

| File | Reason |
|------|--------|
| `webapp/views/api_docs.py` | User requested removal |

---

## Implementation Order

```
Phase 1 (UI Redesign) ─┐
Phase 2 (Remove API)  ─┤── Can run in parallel, no dependencies
                        │
Phase 3 (Cure Options) ─┤── Depends on Phase 1 for styling
                        │
Phase 4 (Patient Profiles) ── Depends on Phase 1; creates storage for Phase 5
                        │
Phase 5 (Book Doctor)  ─┤── Depends on Phase 3 (cancer types) + Phase 4 (storage)
                        │
Phase 6 (Extras)       ─┘── Incremental, after all core features
```

## Verification

1. **Run the app**: `streamlit run webapp/app.py` — verify all 9 pages load, sidebar nav works
2. **UI check**: Confirm light theme, no dark-mode remnants, clean cards, readable text
3. **Cure Options**: Select each cancer type → precautions, treatments, survival rates display
4. **Patient Profiles**: Create → search → view → link to prediction → verify saved
5. **Book Doctor**: Filter by cancer type → see relevant doctors → book → see confirmation
6. **Single Prediction**: Run prediction → cure info shows → "Consult Specialist" button works
7. **Tests**: `pytest tests/ -v` — existing tests still pass
8. **Lint**: `ruff check webapp/` — no new violations

## No New Dependencies

All features use Python stdlib: `sqlite3`, `uuid`, `datetime`, `json`. No pip installs needed.
