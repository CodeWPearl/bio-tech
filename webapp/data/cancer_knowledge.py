"""Structured knowledge base for cancer precautions and treatment options.

Contains curated information for the five TCGA cancer types used by the
predictor, plus pathogenicity-level clinical guidance.  All data is sourced
from publicly available medical literature and guidelines (NCCN, ACS, NCI).
"""

from __future__ import annotations

CANCER_KNOWLEDGE: dict[str, dict] = {
    "Breast Invasive Carcinoma": {
        "abbreviation": "BRCA",
        "overview": (
            "Breast invasive carcinoma is the most common cancer among women "
            "worldwide. It begins in the breast tissue and can spread to "
            "nearby lymph nodes or distant organs. Early detection through "
            "screening significantly improves outcomes. Molecular subtypes "
            "(Luminal A/B, HER2-enriched, Triple-negative) guide treatment "
            "selection."
        ),
        "precautions": [
            {
                "category": "Screening",
                "detail": (
                    "Regular mammography starting at age 40 (annually or "
                    "biannually based on risk). Clinical breast exams every "
                    "1-3 years for women 20-39, annually after 40. Breast "
                    "MRI recommended for high-risk individuals."
                ),
            },
            {
                "category": "Lifestyle",
                "detail": (
                    "Maintain a healthy weight, especially after menopause. "
                    "Engage in regular physical activity (150+ minutes/week). "
                    "Limit alcohol consumption. Avoid prolonged hormone "
                    "replacement therapy when possible."
                ),
            },
            {
                "category": "Genetic",
                "detail": (
                    "BRCA1/BRCA2 genetic testing for individuals with family "
                    "history of breast or ovarian cancer. Genetic counseling "
                    "recommended before and after testing. Consider risk-"
                    "reducing strategies if positive (enhanced surveillance, "
                    "chemoprevention, prophylactic surgery)."
                ),
            },
            {
                "category": "Environmental",
                "detail": (
                    "Minimize unnecessary radiation exposure, especially "
                    "chest radiation during adolescence. Be aware of "
                    "occupational chemical exposures. Maintain a diet rich "
                    "in fruits, vegetables, and whole grains."
                ),
            },
        ],
        "treatment_options": [
            {
                "name": "Surgery",
                "description": (
                    "Lumpectomy (breast-conserving) or mastectomy depending "
                    "on tumor size, location, and patient preference. "
                    "Sentinel lymph node biopsy to assess spread."
                ),
                "stage": "All stages",
            },
            {
                "name": "Radiation Therapy",
                "description": (
                    "External beam radiation or brachytherapy, typically "
                    "after lumpectomy. Reduces local recurrence risk by "
                    "approximately 50%."
                ),
                "stage": "Stage I-III",
            },
            {
                "name": "Chemotherapy",
                "description": (
                    "Neoadjuvant (before surgery) or adjuvant (after surgery) "
                    "systemic chemotherapy. Regimens include AC-T, TC, or "
                    "CMF depending on subtype and stage."
                ),
                "stage": "Stage II+",
            },
            {
                "name": "Hormonal Therapy",
                "description": (
                    "Tamoxifen or aromatase inhibitors for hormone receptor-"
                    "positive tumors. Typically administered for 5-10 years "
                    "post-surgery."
                ),
                "stage": "HR+ tumors, all stages",
            },
            {
                "name": "Targeted Therapy",
                "description": (
                    "Trastuzumab (Herceptin) and pertuzumab for HER2-positive "
                    "tumors. CDK4/6 inhibitors (palbociclib, ribociclib) for "
                    "HR+/HER2- advanced disease. PARP inhibitors for "
                    "BRCA-mutated cancers."
                ),
                "stage": "Based on molecular profile",
            },
            {
                "name": "Immunotherapy",
                "description": (
                    "Pembrolizumab combined with chemotherapy for PD-L1 "
                    "positive triple-negative breast cancer (TNBC). Active "
                    "area of clinical research."
                ),
                "stage": "Advanced TNBC",
            },
        ],
        "survival_rates": {
            "Stage I": "99%",
            "Stage II": "93%",
            "Stage III": "72%",
            "Stage IV": "29%",
        },
        "key_genes": ["BRCA1", "BRCA2", "ERBB2", "TP53", "PIK3CA"],
        "clinical_trials_url": (
            "https://clinicaltrials.gov/search?cond=Breast+Cancer"
        ),
        "nccn_url": "https://www.nccn.org/patients/guidelines/content/PDF/breast-patient.pdf",
    },
    "Lung Adenocarcinoma": {
        "abbreviation": "LUAD",
        "overview": (
            "Lung adenocarcinoma is the most common subtype of non-small cell "
            "lung cancer (NSCLC), accounting for about 40% of all lung "
            "cancers. It typically arises in the outer regions of the lung "
            "and is associated with both smoking and non-smoking patients. "
            "Advances in targeted therapy and immunotherapy have "
            "significantly improved outcomes for specific molecular subtypes."
        ),
        "precautions": [
            {
                "category": "Screening",
                "detail": (
                    "Annual low-dose CT (LDCT) screening for adults 50-80 "
                    "with a 20+ pack-year smoking history who currently smoke "
                    "or quit within the past 15 years. Early detection can "
                    "reduce lung cancer mortality by 20%."
                ),
            },
            {
                "category": "Lifestyle",
                "detail": (
                    "Smoking cessation is the single most effective "
                    "prevention measure. Avoid secondhand smoke exposure. "
                    "Regular exercise and a balanced diet support lung "
                    "health. Radon testing for homes in high-risk areas."
                ),
            },
            {
                "category": "Genetic",
                "detail": (
                    "Molecular profiling of tumor tissue for actionable "
                    "mutations (EGFR, ALK, ROS1, BRAF, KRAS G12C, MET, "
                    "RET, NTRK). Liquid biopsy as an alternative for "
                    "tissue-unavailable cases."
                ),
            },
            {
                "category": "Environmental",
                "detail": (
                    "Test home for radon and mitigate if levels exceed "
                    "4 pCi/L. Use proper protective equipment for "
                    "occupational exposures (asbestos, diesel exhaust, "
                    "certain chemicals). Monitor air quality indices."
                ),
            },
        ],
        "treatment_options": [
            {
                "name": "Surgery",
                "description": (
                    "Lobectomy (preferred), segmentectomy, or wedge resection "
                    "for early-stage disease. Video-assisted thoracoscopic "
                    "surgery (VATS) or robotic approaches minimize recovery."
                ),
                "stage": "Stage I-II",
            },
            {
                "name": "Radiation Therapy",
                "description": (
                    "Stereotactic body radiation therapy (SBRT) for early-"
                    "stage patients who cannot undergo surgery. Concurrent "
                    "chemoradiation for locally advanced (Stage III) disease."
                ),
                "stage": "Stage I-III",
            },
            {
                "name": "Chemotherapy",
                "description": (
                    "Platinum-based doublets (cisplatin/carboplatin + "
                    "pemetrexed) as standard first-line. Adjuvant "
                    "chemotherapy for resected Stage II-III."
                ),
                "stage": "Stage II+",
            },
            {
                "name": "Targeted Therapy",
                "description": (
                    "EGFR inhibitors (osimertinib) for EGFR-mutant tumors. "
                    "ALK inhibitors (alectinib, lorlatinib) for ALK-"
                    "rearranged tumors. KRAS G12C inhibitors (sotorasib, "
                    "adagrasib). Additional targets: ROS1, BRAF V600E, "
                    "MET, RET, NTRK fusions."
                ),
                "stage": "Based on molecular profile",
            },
            {
                "name": "Immunotherapy",
                "description": (
                    "Pembrolizumab, nivolumab, or atezolizumab as single "
                    "agent (PD-L1 ≥50%) or combined with chemotherapy. "
                    "Durvalumab as consolidation after chemoradiation for "
                    "Stage III. Neoadjuvant nivolumab + chemotherapy for "
                    "resectable tumors."
                ),
                "stage": "Stage III-IV",
            },
        ],
        "survival_rates": {
            "Stage I": "92%",
            "Stage II": "60%",
            "Stage III": "36%",
            "Stage IV": "8%",
        },
        "key_genes": ["EGFR", "KRAS", "ALK", "TP53", "STK11"],
        "clinical_trials_url": (
            "https://clinicaltrials.gov/search?cond=Lung+Adenocarcinoma"
        ),
        "nccn_url": "https://www.nccn.org/patients/guidelines/content/PDF/lung-patient.pdf",
    },
    "Colorectal Adenocarcinoma": {
        "abbreviation": "COAD",
        "overview": (
            "Colorectal adenocarcinoma is the third most common cancer "
            "globally. It develops from the glandular cells lining the colon "
            "or rectum, often progressing from precancerous polyps over "
            "10-15 years. Screening colonoscopy can prevent cancer by "
            "detecting and removing polyps. Microsatellite instability (MSI) "
            "status is crucial for treatment planning."
        ),
        "precautions": [
            {
                "category": "Screening",
                "detail": (
                    "Colonoscopy every 10 years starting at age 45 for "
                    "average-risk adults. Earlier and more frequent screening "
                    "for high-risk individuals (family history, IBD, Lynch "
                    "syndrome). Alternative: annual FIT/FOBT or stool DNA "
                    "test every 3 years."
                ),
            },
            {
                "category": "Lifestyle",
                "detail": (
                    "High-fiber diet with fruits, vegetables, and whole "
                    "grains. Limit red and processed meat consumption. "
                    "Regular physical activity (30+ min/day). Maintain "
                    "healthy weight. Limit alcohol and avoid tobacco."
                ),
            },
            {
                "category": "Genetic",
                "detail": (
                    "Lynch syndrome screening (MLH1, MSH2, MSH6, PMS2 "
                    "genes) for individuals with strong family history or "
                    "early-onset colorectal cancer. Familial adenomatous "
                    "polyposis (FAP/APC gene) testing when indicated."
                ),
            },
            {
                "category": "Environmental",
                "detail": (
                    "Aspirin may reduce colorectal cancer risk in certain "
                    "populations (discuss with physician). Adequate vitamin D "
                    "and calcium intake may have protective effects. Avoid "
                    "chronic inflammatory conditions of the bowel."
                ),
            },
        ],
        "treatment_options": [
            {
                "name": "Surgery",
                "description": (
                    "Colectomy with lymph node dissection for colon cancer. "
                    "Total mesorectal excision (TME) for rectal cancer. "
                    "Minimally invasive (laparoscopic/robotic) approaches "
                    "preferred when feasible."
                ),
                "stage": "All stages",
            },
            {
                "name": "Radiation Therapy",
                "description": (
                    "Primarily used for rectal cancer — neoadjuvant "
                    "chemoradiation to shrink tumors before surgery. "
                    "Short-course radiation (5 x 5 Gy) or long-course "
                    "chemoradiation protocols."
                ),
                "stage": "Stage II-III (rectal)",
            },
            {
                "name": "Chemotherapy",
                "description": (
                    "FOLFOX (5-FU + leucovorin + oxaliplatin) or CAPOX "
                    "(capecitabine + oxaliplatin) as adjuvant therapy. "
                    "FOLFIRI or FOLFOXIRI for advanced disease."
                ),
                "stage": "Stage III+",
            },
            {
                "name": "Targeted Therapy",
                "description": (
                    "Anti-VEGF agents (bevacizumab) or anti-EGFR agents "
                    "(cetuximab, panitumumab) for RAS wild-type tumors. "
                    "Encorafenib + cetuximab for BRAF V600E-mutant tumors."
                ),
                "stage": "Stage IV",
            },
            {
                "name": "Immunotherapy",
                "description": (
                    "Pembrolizumab or nivolumab + ipilimumab for MSI-high/"
                    "dMMR tumors. First-line immunotherapy now standard for "
                    "MSI-H metastatic CRC — often superior to chemotherapy."
                ),
                "stage": "MSI-H/dMMR tumors",
            },
        ],
        "survival_rates": {
            "Stage I": "91%",
            "Stage II": "82%",
            "Stage III": "71%",
            "Stage IV": "14%",
        },
        "key_genes": ["APC", "KRAS", "TP53", "PIK3CA", "BRAF"],
        "clinical_trials_url": (
            "https://clinicaltrials.gov/search?cond=Colorectal+Cancer"
        ),
        "nccn_url": "https://www.nccn.org/patients/guidelines/content/PDF/colon-patient.pdf",
    },
    "Uterine Corpus Endometrial Carcinoma": {
        "abbreviation": "UCEC",
        "overview": (
            "Uterine corpus endometrial carcinoma is the most common "
            "gynecological cancer in developed countries. It originates from "
            "the endometrial lining of the uterus and is often detected "
            "early due to abnormal uterine bleeding. Molecular "
            "classification (POLE ultramutated, MSI-H, copy-number low, "
            "copy-number high) increasingly guides prognosis and treatment."
        ),
        "precautions": [
            {
                "category": "Screening",
                "detail": (
                    "No routine screening for average-risk women. Prompt "
                    "evaluation of any abnormal uterine bleeding, especially "
                    "postmenopausal bleeding. Transvaginal ultrasound and "
                    "endometrial biopsy for symptomatic patients. Annual "
                    "screening starting at age 35 for Lynch syndrome carriers."
                ),
            },
            {
                "category": "Lifestyle",
                "detail": (
                    "Maintain a healthy weight — obesity is the strongest "
                    "modifiable risk factor (2-4x increased risk). Regular "
                    "physical activity. Manage diabetes and metabolic "
                    "syndrome. Consider oral contraceptive use (protective "
                    "effect with long-term use)."
                ),
            },
            {
                "category": "Genetic",
                "detail": (
                    "Lynch syndrome testing (MLH1, MSH2, MSH6, PMS2) for "
                    "patients with family history or early-onset disease. "
                    "Cowden syndrome (PTEN) screening when indicated. "
                    "Genetic counseling for hereditary predisposition "
                    "syndromes."
                ),
            },
            {
                "category": "Environmental",
                "detail": (
                    "Monitor and manage unopposed estrogen exposure "
                    "(estrogen-only HRT, tamoxifen use for breast cancer). "
                    "Progesterone co-administration with estrogen HRT "
                    "reduces risk."
                ),
            },
        ],
        "treatment_options": [
            {
                "name": "Surgery",
                "description": (
                    "Total hysterectomy with bilateral salpingo-oophorectomy "
                    "(TH-BSO) as primary treatment. Sentinel lymph node "
                    "mapping increasingly used instead of full "
                    "lymphadenectomy. Minimally invasive surgery preferred."
                ),
                "stage": "All stages",
            },
            {
                "name": "Radiation Therapy",
                "description": (
                    "Vaginal cuff brachytherapy for intermediate-risk. "
                    "External beam pelvic radiation for high-risk features. "
                    "Adjuvant radiation reduces local recurrence."
                ),
                "stage": "Stage I (high-risk) to III",
            },
            {
                "name": "Chemotherapy",
                "description": (
                    "Carboplatin + paclitaxel as standard adjuvant regimen "
                    "for advanced or high-risk disease. Combined with "
                    "radiation for Stage III-IV."
                ),
                "stage": "Stage III+",
            },
            {
                "name": "Hormonal Therapy",
                "description": (
                    "Progestins (medroxyprogesterone, megestrol) for "
                    "low-grade, hormone receptor-positive tumors. May be "
                    "used for fertility preservation in young patients with "
                    "early-stage disease."
                ),
                "stage": "Low-grade, early stage",
            },
            {
                "name": "Immunotherapy",
                "description": (
                    "Pembrolizumab + lenvatinib for non-MSI-H advanced "
                    "endometrial cancer. Dostarlimab for dMMR/MSI-H "
                    "recurrent or advanced disease."
                ),
                "stage": "Advanced/recurrent",
            },
        ],
        "survival_rates": {
            "Stage I": "95%",
            "Stage II": "77%",
            "Stage III": "57%",
            "Stage IV": "17%",
        },
        "key_genes": ["PTEN", "PIK3CA", "ARID1A", "TP53", "CTNNB1"],
        "clinical_trials_url": (
            "https://clinicaltrials.gov/search?cond=Endometrial+Cancer"
        ),
        "nccn_url": "https://www.nccn.org/patients/guidelines/content/PDF/uterine-patient.pdf",
    },
    "Ovarian Serous Cystadenocarcinoma": {
        "abbreviation": "OV",
        "overview": (
            "Ovarian serous cystadenocarcinoma (high-grade serous carcinoma) "
            "is the most common and aggressive subtype of ovarian cancer, "
            "accounting for approximately 70% of ovarian cancer deaths. It "
            "is often diagnosed at advanced stages due to nonspecific "
            "symptoms. BRCA1/2 mutations and homologous recombination "
            "deficiency (HRD) status are critical for treatment decisions."
        ),
        "precautions": [
            {
                "category": "Screening",
                "detail": (
                    "No effective routine screening for average-risk women. "
                    "CA-125 blood test and transvaginal ultrasound for "
                    "high-risk individuals (BRCA carriers, family history). "
                    "Awareness of symptoms: bloating, pelvic pain, urinary "
                    "urgency, early satiety lasting >2 weeks."
                ),
            },
            {
                "category": "Lifestyle",
                "detail": (
                    "Oral contraceptive use reduces risk by 30-50% with "
                    "5+ years of use. Pregnancy and breastfeeding are "
                    "protective. Maintain healthy weight. Tubal ligation "
                    "may reduce risk."
                ),
            },
            {
                "category": "Genetic",
                "detail": (
                    "BRCA1/BRCA2 testing recommended for all ovarian cancer "
                    "patients and high-risk relatives. RAD51C, RAD51D, "
                    "BRIP1 testing for hereditary risk. Risk-reducing "
                    "salpingo-oophorectomy (RRSO) for BRCA carriers "
                    "typically recommended by age 35-40."
                ),
            },
            {
                "category": "Environmental",
                "detail": (
                    "Avoid talcum powder use in the genital area (debated "
                    "but precautionary). Awareness of endometriosis as a "
                    "risk factor for certain subtypes."
                ),
            },
        ],
        "treatment_options": [
            {
                "name": "Surgery",
                "description": (
                    "Primary debulking surgery (cytoreduction) aiming for "
                    "no visible residual disease. Includes TH-BSO, "
                    "omentectomy, and peritoneal biopsies. Interval "
                    "debulking after neoadjuvant chemotherapy for advanced "
                    "cases."
                ),
                "stage": "All stages",
            },
            {
                "name": "Chemotherapy",
                "description": (
                    "Carboplatin + paclitaxel as first-line standard (6 "
                    "cycles). Dose-dense weekly paclitaxel may improve "
                    "outcomes. Intraperitoneal (IP) chemotherapy for "
                    "optimally debulked Stage III."
                ),
                "stage": "Stage I (high-grade) to IV",
            },
            {
                "name": "Targeted Therapy",
                "description": (
                    "Bevacizumab (anti-VEGF) added to first-line "
                    "chemotherapy and as maintenance. PARP inhibitors "
                    "(olaparib, niraparib, rucaparib) as maintenance — "
                    "especially effective for BRCA-mutated and HRD-positive "
                    "tumors."
                ),
                "stage": "Stage III-IV",
            },
            {
                "name": "Immunotherapy",
                "description": (
                    "Emerging role — clinical trials investigating "
                    "checkpoint inhibitors in combination with chemotherapy "
                    "or PARP inhibitors. Currently not standard first-line "
                    "but promising in recurrent disease."
                ),
                "stage": "Clinical trials",
            },
            {
                "name": "Hormonal Therapy",
                "description": (
                    "Tamoxifen or aromatase inhibitors for recurrent "
                    "low-grade serous carcinoma. Limited role in high-grade "
                    "serous histology."
                ),
                "stage": "Recurrent low-grade",
            },
        ],
        "survival_rates": {
            "Stage I": "93%",
            "Stage II": "70%",
            "Stage III": "39%",
            "Stage IV": "17%",
        },
        "key_genes": ["BRCA1", "BRCA2", "TP53", "NF1", "RB1"],
        "clinical_trials_url": (
            "https://clinicaltrials.gov/search?cond=Ovarian+Cancer"
        ),
        "nccn_url": "https://www.nccn.org/patients/guidelines/content/PDF/ovarian-patient.pdf",
    },
}


PATHOGENICITY_GUIDANCE: dict[str, dict] = {
    "Pathogenic": {
        "severity": "High",
        "severity_color": "#EF4444",
        "recommendation": (
            "This variant is classified as pathogenic. Consider genetic "
            "counseling and specialist referral for comprehensive evaluation "
            "and management planning."
        ),
        "follow_up": [
            "Confirmatory testing with an independent method",
            "Family member screening for the same variant",
            "Specialist consultation (oncologist/geneticist)",
            "Discussion of risk-reducing strategies",
            "Enrollment in appropriate surveillance programs",
        ],
    },
    "Likely Pathogenic": {
        "severity": "Moderate-High",
        "severity_color": "#F97316",
        "recommendation": (
            "This variant is classified as likely pathogenic. Further "
            "evaluation and genetic counseling are recommended to confirm "
            "clinical significance and guide management."
        ),
        "follow_up": [
            "Additional functional or segregation studies",
            "Genetic counseling for the patient and family",
            "Enhanced surveillance based on cancer type",
            "Periodic re-evaluation as new evidence emerges",
        ],
    },
    "Benign": {
        "severity": "Low",
        "severity_color": "#10B981",
        "recommendation": (
            "This variant is classified as benign. It is not expected to "
            "increase cancer risk. Standard age-appropriate screening "
            "guidelines apply."
        ),
        "follow_up": [
            "Continue standard screening per guidelines",
            "No additional genetic workup required for this variant",
        ],
    },
    "Likely Benign": {
        "severity": "Low",
        "severity_color": "#34D399",
        "recommendation": (
            "This variant is classified as likely benign. It is unlikely "
            "to be clinically significant. Routine screening is recommended."
        ),
        "follow_up": [
            "Continue routine screening per guidelines",
            "Periodic re-classification check as databases update",
        ],
    },
}


def get_cancer_info(cancer_type: str) -> dict | None:
    """Return knowledge base entry for the given cancer type, or None."""
    return CANCER_KNOWLEDGE.get(cancer_type)


def get_pathogenicity_guidance(predicted_class: str) -> dict | None:
    """Return clinical guidance for the given pathogenicity class."""
    return PATHOGENICITY_GUIDANCE.get(predicted_class)


def list_cancer_types() -> list[str]:
    """Return sorted list of all cancer types in the knowledge base."""
    return sorted(CANCER_KNOWLEDGE.keys())
