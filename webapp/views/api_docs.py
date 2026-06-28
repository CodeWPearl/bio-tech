"""API Documentation page — interactive reference with examples."""

from __future__ import annotations

import streamlit as st

from webapp.utils.api_client import APIClient

ENDPOINTS = [
    {
        "method": "GET",
        "path": "/health",
        "desc": "Check API health status and model availability",
        "curl": "curl -s http://localhost:8001/health | python -m json.tool",
        "python": (
            "import requests\n\n"
            'resp = requests.get("http://localhost:8001/health")\n'
            "print(resp.json())"
        ),
        "response": (
            '{\n  "status": "healthy",\n  "model_loaded": true,\n'
            '  "version": "1.0.0"\n}'
        ),
    },
    {
        "method": "GET",
        "path": "/version",
        "desc": "Get API version and build information",
        "curl": "curl -s http://localhost:8001/version | python -m json.tool",
        "python": (
            "import requests\n\n"
            'resp = requests.get("http://localhost:8001/version")\n'
            "print(resp.json())"
        ),
        "response": '{\n  "version": "1.0.0",\n  "build": "2026-01-01"\n}',
    },
    {
        "method": "POST",
        "path": "/predict",
        "desc": "Predict pathogenicity for a single variant",
        "curl": (
            "curl -X POST http://localhost:8001/predict \\\n"
            '  -H "Content-Type: application/json" \\\n'
            "  -d '{\n"
            '    "gene_symbol": "BRCA1",\n'
            '    "mutation_type": "Missense_Mutation",\n'
            '    "chromosome": "17",\n'
            '    "start_position": 43044295,\n'
            '    "reference_allele": "A",\n'
            '    "variant_allele": "T",\n'
            '    "protein_change": "p.C61G",\n'
            '    "include_explanation": true,\n'
            '    "include_uncertainty": true\n'
            "  }'"
        ),
        "python": (
            "import requests\n\n"
            "payload = {\n"
            '    "gene_symbol": "BRCA1",\n'
            '    "mutation_type": "Missense_Mutation",\n'
            '    "chromosome": "17",\n'
            '    "start_position": 43044295,\n'
            '    "reference_allele": "A",\n'
            '    "variant_allele": "T",\n'
            '    "protein_change": "p.C61G",\n'
            '    "include_explanation": True,\n'
            '    "include_uncertainty": True,\n'
            "}\n\n"
            'resp = requests.post("http://localhost:8001/predict", json=payload)\n'
            "result = resp.json()\n"
            "print(f\"Class: {result['predicted_class']}\")\n"
            "print(f\"Confidence: {result['confidence']:.2%}\")"
        ),
        "response": (
            "{\n"
            '  "variant_id": "BRCA1_17_43044295_A_T",\n'
            '  "predicted_class": "Pathogenic",\n'
            '  "confidence": 0.934,\n'
            '  "class_probabilities": {\n'
            '    "Pathogenic": 0.934,\n'
            '    "Likely Pathogenic": 0.041,\n'
            '    "Benign": 0.015,\n'
            '    "Likely Benign": 0.010\n'
            "  },\n"
            '  "recommendation": "High Confidence"\n'
            "}"
        ),
    },
    {
        "method": "POST",
        "path": "/predict/batch",
        "desc": "Predict pathogenicity for up to 100 variants",
        "curl": (
            "curl -X POST http://localhost:8001/predict/batch \\\n"
            '  -H "Content-Type: application/json" \\\n'
            "  -d '{\n"
            '    "variants": [\n'
            '      {"gene_symbol": "BRCA1", "mutation_type": "Missense_Mutation",\n'
            '       "chromosome": "17", "start_position": 43044295,\n'
            '       "reference_allele": "A", "variant_allele": "T"},\n'
            '      {"gene_symbol": "TP53", "mutation_type": "Nonsense_Mutation",\n'
            '       "chromosome": "17", "start_position": 7577538,\n'
            '       "reference_allele": "C", "variant_allele": "T"}\n'
            "    ]\n"
            "  }'"
        ),
        "python": (
            "import requests\n\n"
            "variants = [\n"
            "    {\n"
            '        "gene_symbol": "BRCA1",\n'
            '        "mutation_type": "Missense_Mutation",\n'
            '        "chromosome": "17",\n'
            '        "start_position": 43044295,\n'
            '        "reference_allele": "A",\n'
            '        "variant_allele": "T",\n'
            "    },\n"
            "    {\n"
            '        "gene_symbol": "TP53",\n'
            '        "mutation_type": "Nonsense_Mutation",\n'
            '        "chromosome": "17",\n'
            '        "start_position": 7577538,\n'
            '        "reference_allele": "C",\n'
            '        "variant_allele": "T",\n'
            "    },\n"
            "]\n\n"
            "resp = requests.post(\n"
            '    "http://localhost:8001/predict/batch",\n'
            '    json={"variants": variants},\n'
            ")\n"
            "data = resp.json()\n"
            "for pred in data['predictions']:\n"
            "    print(f\"{pred['variant_id']}: {pred['predicted_class']}\")"
        ),
        "response": (
            "{\n"
            '  "predictions": [\n'
            '    {"variant_id": "BRCA1_...", "predicted_class": "Pathogenic", ...},\n'
            '    {"variant_id": "TP53_...", "predicted_class": "Pathogenic", ...}\n'
            "  ],\n"
            '  "summary": {\n'
            '    "total_variants": 2,\n'
            '    "class_counts": {"Pathogenic": 2},\n'
            '    "average_confidence": 0.891\n'
            "  }\n"
            "}"
        ),
    },
    {
        "method": "GET",
        "path": "/genes",
        "desc": "List all genes with variant counts and cancer driver status",
        "curl": "curl -s http://localhost:8001/genes | python -m json.tool",
        "python": (
            "import requests\n\n"
            'genes = requests.get("http://localhost:8001/genes").json()\n'
            "for gene in genes[:5]:\n"
            "    print(f\"{gene['gene_symbol']}: {gene['variant_count']} variants\")"
        ),
        "response": (
            "[\n"
            '  {"gene_symbol": "TP53", "variant_count": 245, '
            '"is_cancer_driver": true},\n'
            '  {"gene_symbol": "BRCA1", "variant_count": 189, '
            '"is_cancer_driver": true},\n'
            "  ...\n"
            "]"
        ),
    },
    {
        "method": "GET",
        "path": "/genes/{symbol}",
        "desc": "Get detailed information for a specific gene",
        "curl": "curl -s http://localhost:8001/genes/BRCA1 | python -m json.tool",
        "python": (
            "import requests\n\n"
            'gene = requests.get("http://localhost:8001/genes/BRCA1").json()\n'
            "print(f\"Gene: {gene['gene_symbol']}\")\n"
            "print(f\"Variants: {gene['variant_count']}\")\n"
            "print(f\"Driver: {gene['is_known_cancer_driver']}\")"
        ),
        "response": (
            "{\n"
            '  "gene_symbol": "BRCA1",\n'
            '  "variant_count": 189,\n'
            '  "is_known_cancer_driver": true,\n'
            '  "cosmic_census_info": "TSG, breast/ovarian cancer",\n'
            '  "class_distribution": {"Pathogenic": 120, "Benign": 45, ...}\n'
            "}"
        ),
    },
    {
        "method": "GET",
        "path": "/stats",
        "desc": "Get dataset statistics (class distribution, gene counts)",
        "curl": "curl -s http://localhost:8001/stats | python -m json.tool",
        "python": (
            "import requests\n\n"
            'stats = requests.get("http://localhost:8001/stats").json()\n'
            "print(f\"Total variants: {stats['total_variants']}\")\n"
            "print(f\"Classes: {stats['class_distribution']}\")"
        ),
        "response": (
            "{\n"
            '  "total_variants": 15234,\n'
            '  "class_distribution": {"Pathogenic": 4521, "Benign": 5123, ...},\n'
            '  "cancer_types": ["BRCA", "LUAD", ...],\n'
            '  "top_genes": [{"TP53": 245}, {"BRCA1": 189}, ...]\n'
            "}"
        ),
    },
    {
        "method": "GET",
        "path": "/model/info",
        "desc": "Get model architecture and training details",
        "curl": "curl -s http://localhost:8001/model/info | python -m json.tool",
        "python": (
            "import requests\n\n"
            'info = requests.get("http://localhost:8001/model/info").json()\n'
            "print(f\"Architecture: {info['architecture']}\")\n"
            "print(f\"Parameters: {info['total_parameters']:,}\")"
        ),
        "response": (
            "{\n"
            '  "architecture": "Multi-Omics DL",\n'
            '  "fusion_type": "Cross-Attention",\n'
            '  "total_parameters": 2847923,\n'
            '  "encoder_parameters": {"mutation": 512000, ...}\n'
            "}"
        ),
    },
    {
        "method": "POST",
        "path": "/explain/shap",
        "desc": "Get SHAP explanations for a variant prediction",
        "curl": (
            "curl -X POST http://localhost:8001/explain/shap \\\n"
            '  -H "Content-Type: application/json" \\\n'
            "  -d '{\"gene_symbol\": \"BRCA1\", \"mutation_type\": "
            '"Missense_Mutation",\n'
            '       "chromosome": "17", "start_position": 43044295,\n'
            '       "reference_allele": "A", "variant_allele": "T"}\''
        ),
        "python": (
            "import requests\n\n"
            "payload = {\n"
            '    "gene_symbol": "BRCA1",\n'
            '    "mutation_type": "Missense_Mutation",\n'
            '    "chromosome": "17",\n'
            '    "start_position": 43044295,\n'
            '    "reference_allele": "A",\n'
            '    "variant_allele": "T",\n'
            "}\n"
            "resp = requests.post(\n"
            '    "http://localhost:8001/explain/shap", json=payload\n'
            ")\n"
            "shap_data = resp.json()\n"
            "print(shap_data['modality_contributions'])"
        ),
        "response": (
            "{\n"
            '  "modality_contributions": {\n'
            '    "mutation": 0.45, "expression": 0.30, "methylation": 0.15\n'
            "  },\n"
            '  "top_positive_features": [...],\n'
            '  "top_negative_features": [...]\n'
            "}"
        ),
    },
    {
        "method": "POST",
        "path": "/explain/attention",
        "desc": "Get cross-modal attention weights",
        "curl": (
            "curl -X POST http://localhost:8001/explain/attention \\\n"
            '  -H "Content-Type: application/json" \\\n'
            "  -d '{\"gene_symbol\": \"TP53\", \"mutation_type\": "
            '"Nonsense_Mutation",\n'
            '       "chromosome": "17", "start_position": 7577538,\n'
            '       "reference_allele": "C", "variant_allele": "T"}\''
        ),
        "python": (
            "import requests\n\n"
            "payload = {\n"
            '    "gene_symbol": "TP53",\n'
            '    "mutation_type": "Nonsense_Mutation",\n'
            '    "chromosome": "17",\n'
            '    "start_position": 7577538,\n'
            '    "reference_allele": "C",\n'
            '    "variant_allele": "T",\n'
            "}\n"
            "resp = requests.post(\n"
            '    "http://localhost:8001/explain/attention", json=payload\n'
            ")\n"
            "print(resp.json())"
        ),
        "response": (
            "{\n"
            '  "attention_weights": {\n'
            '    "mutation": 0.42, "expression": 0.33, "methylation": 0.25\n'
            "  }\n"
            "}"
        ),
    },
    {
        "method": "GET",
        "path": "/explain/global",
        "desc": "Get global feature importance across the model",
        "curl": (
            "curl -s http://localhost:8001/explain/global | python -m json.tool"
        ),
        "python": (
            "import requests\n\n"
            "global_exp = requests.get(\n"
            '    "http://localhost:8001/explain/global"\n'
            ").json()\n"
            "for feat in global_exp['top_features'][:5]:\n"
            "    print(f\"{feat['name']}: {feat['importance']:.4f}\")"
        ),
        "response": (
            "{\n"
            '  "modality_importance": {\n'
            '    "mutation": 0.42, "expression": 0.31, "methylation": 0.15\n'
            "  },\n"
            '  "top_features": [\n'
            '    {"name": "mutation_type_encoded", "importance": 0.089},\n'
            '    {"name": "tp53_expression", "importance": 0.067}\n'
            "  ]\n"
            "}"
        ),
    },
]


def render(client: APIClient) -> None:
    """Render the API Documentation page."""
    st.markdown(
        """
        <div class="dashboard-header">
            <h1>\U0001f4bb API Documentation</h1>
            <p>Interactive reference for the Cancer Mutation Pathogenicity
            Predictor REST API</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Swagger link ---
    st.markdown(
        """
        <div class="glass-card" style="text-align:center;padding:1.5rem">
            <p style="margin:0;font-size:1.1rem">
                <strong style="color:#A5B4FC !important">FastAPI Swagger UI
                </strong> — auto-generated interactive docs
            </p>
            <p style="margin:0.5rem 0 0;color:#94A3B8 !important">
                <code style="color:#A5B4FC">http://localhost:8001/docs</code>
                &nbsp;|&nbsp;
                <code style="color:#A5B4FC">http://localhost:8001/redoc</code>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Base URL ---
    st.markdown(
        """
        <div class="glass-card">
            <h3 style="margin-top:0">Base URL</h3>
            <code style="font-size:1.1rem;color:#A5B4FC">
                http://localhost:8001</code>
            <p style="color:#94A3B8 !important;margin:0.5rem 0 0;
               font-size:0.85rem">
                All endpoints accept and return JSON. Set the
                <code style="color:#A5B4FC">Content-Type: application/json</code>
                header for POST requests.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Endpoint list ---
    st.markdown("---")
    st.subheader("Endpoints")

    for ep in ENDPOINTS:
        method = ep["method"]
        path = ep["path"]
        method_color = "#10B981" if method == "GET" else "#6366F1"
        badge = (
            f'<span style="background:{method_color}22;color:{method_color};'
            f"padding:2px 10px;border-radius:6px;font-size:0.8rem;"
            f'font-weight:700;font-family:monospace">{method}</span>'
        )

        with st.expander(f"{method}  {path} — {ep['desc']}"):
            st.markdown(
                f"{badge}&nbsp;&nbsp;"
                f'<code style="color:#A5B4FC;font-size:1rem">{path}</code>',
                unsafe_allow_html=True,
            )
            st.markdown(f"**{ep['desc']}**")

            tab_curl, tab_py, tab_resp = st.tabs(
                ["cURL", "Python", "Response"]
            )
            with tab_curl:
                st.code(ep["curl"], language="bash")
            with tab_py:
                st.code(ep["python"], language="python")
            with tab_resp:
                st.code(ep["response"], language="json")

    st.markdown("---")

    # --- Quick-start Python snippet ---
    st.markdown(
        '<div class="glass-card"><h3 style="margin-top:0">'
        '\U0001f40d Python Quick Start</h3></div>',
        unsafe_allow_html=True,
    )
    st.code(
        'import requests\n\n'
        'BASE_URL = "http://localhost:8001"\n\n'
        "# 1. Check health\n"
        'health = requests.get(f"{BASE_URL}/health").json()\n'
        "assert health['status'] == 'healthy'\n\n"
        "# 2. Single prediction\n"
        "variant = {\n"
        '    "gene_symbol": "BRAF",\n'
        '    "mutation_type": "Missense_Mutation",\n'
        '    "chromosome": "7",\n'
        '    "start_position": 140753336,\n'
        '    "reference_allele": "A",\n'
        '    "variant_allele": "T",\n'
        '    "protein_change": "p.V600E",\n'
        '    "include_explanation": True,\n'
        '    "include_uncertainty": True,\n'
        "}\n"
        'result = requests.post(f"{BASE_URL}/predict", json=variant).json()\n'
        "print(f\"Prediction: {result['predicted_class']}\")\n"
        "print(f\"Confidence: {result['confidence']:.1%}\")\n\n"
        "# 3. Batch prediction\n"
        "batch = {\n"
        '    "variants": [\n'
        "        variant,\n"
        '        {**variant, "gene_symbol": "TP53", "chromosome": "17",\n'
        '         "start_position": 7577538, "reference_allele": "C"},\n'
        "    ]\n"
        "}\n"
        'batch_result = requests.post(f"{BASE_URL}/predict/batch", '
        "json=batch).json()\n"
        "for pred in batch_result['predictions']:\n"
        "    print(f\"{pred['variant_id']}: {pred['predicted_class']}\")",
        language="python",
    )

    # --- Error codes ---
    st.markdown("---")
    st.markdown(
        '<div class="glass-card"><h3 style="margin-top:0">'
        '\U0001f6a8 Error Codes</h3></div>',
        unsafe_allow_html=True,
    )
    error_data = [
        ("200", "OK", "Request succeeded"),
        ("422", "Validation Error", "Invalid input — check required fields"),
        ("404", "Not Found", "Resource not found (e.g. unknown gene)"),
        ("500", "Server Error", "Internal error — check API logs"),
    ]
    error_html = ""
    for code, name, desc in error_data:
        code_color = (
            "#10B981" if code == "200"
            else "#F59E0B" if code in ("404", "422")
            else "#EF4444"
        )
        error_html += (
            f'<div style="display:flex;align-items:center;gap:12px;'
            f"padding:0.5rem 0;border-bottom:1px solid "
            f'rgba(99,102,241,0.08)">'
            f'<span style="background:{code_color}22;color:{code_color};'
            f"padding:2px 10px;border-radius:6px;font-size:0.8rem;"
            f'font-weight:700;font-family:monospace;min-width:40px;'
            f'text-align:center">{code}</span>'
            f'<span style="color:#F1F5F9;font-weight:600;'
            f'font-size:0.85rem">{name}</span>'
            f'<span style="color:#64748B;font-size:0.85rem;'
            f'margin-left:auto">{desc}</span>'
            f"</div>"
        )
    st.markdown(error_html, unsafe_allow_html=True)

    st.markdown("---")
    st.caption(
        "Cancer Mutation Pathogenicity Predictor API  ·  "
        "FastAPI  ·  OpenAPI 3.0"
    )
