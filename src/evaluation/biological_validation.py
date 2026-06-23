"""Biological validation of pathogenicity predictions.

Cross-references model predictions with the COSMIC Cancer Gene Census,
ClinVar review confidence, and per-gene accuracy analysis to assess
biological plausibility of the trained model.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

logger = logging.getLogger(__name__)

COSMIC_CENSUS_GENES: frozenset[str] = frozenset({
    "ABL1", "ABL2", "ACKR3", "ACSL3", "ACSL6", "ACVR1", "ACVR2A",
    "AFDN", "AFF1", "AFF3", "AFF4", "AJUBA", "AKT1", "AKT2", "AKT3",
    "ALDH2", "ALK", "AMER1", "ANK1", "APC", "APOBEC3B", "AR", "ARAF",
    "ARHGAP26", "ARHGAP35", "ARHGEF10", "ARHGEF10L", "ARHGEF12",
    "ARID1A", "ARID1B", "ARID2", "ARNT", "ASPSCR1", "ASXL1", "ASXL2",
    "ATF1", "ATIC", "ATM", "ATP1A1", "ATP2B3", "ATR", "ATRX", "AXIN1",
    "AXIN2", "B2M", "BAP1", "BARD1", "BAX", "BCL10", "BCL11A",
    "BCL11B", "BCL2", "BCL2L12", "BCL3", "BCL6", "BCL7A", "BCL9",
    "BCL9L", "BCOR", "BCORL1", "BCR", "BIRC3", "BIRC6", "BLM",
    "BMPR1A", "BRAF", "BRCA1", "BRCA2", "BRD3", "BRD4", "BRIP1",
    "BTG1", "BTK", "BUB1B", "CACNA1D", "CALR", "CAMTA1", "CARD11",
    "CARS1", "CASP8", "CBFA2T3", "CBFB", "CBL", "CBLB", "CBLC",
    "CCDC6", "CCNB1IP1", "CCND1", "CCND2", "CCND3", "CCNE1", "CD274",
    "CD28", "CD74", "CD79A", "CD79B", "CDC73", "CDH1", "CDH10",
    "CDH11", "CDK12", "CDK4", "CDK6", "CDKN1A", "CDKN1B", "CDKN2A",
    "CDKN2B", "CDKN2C", "CEBPA", "CHD4", "CHEK2", "CHIC2", "CHN1",
    "CIITA", "CIC", "CLIP1", "CLTC", "CLTCL1", "CNBP", "CNOT3",
    "CNTRL", "COL1A1", "COL2A1", "CPS1", "CREB1", "CREB3L1",
    "CREB3L2", "CREBBP", "CRLF2", "CRNKL1", "CRTC1", "CRTC3", "CSF1",
    "CSF1R", "CSF3R", "CSMD3", "CTCF", "CTNNA1", "CTNNB1", "CUL3",
    "CUX1", "CXCR4", "CYLD", "CYP2C8", "DAXX", "DCAF12L2", "DCC",
    "DCTN1", "DDB2", "DDIT3", "DDR2", "DDX10", "DDX3X", "DDX41",
    "DDX5", "DDX6", "DEK", "DGCR8", "DICER1", "DIS3", "DNAJB1",
    "DNM2", "DNMT3A", "DROSHA", "DUX4L1", "EBF1", "ECT2L", "EED",
    "EGFR", "EIF1AX", "EIF3E", "EIF4A2", "ELF3", "ELF4", "ELK4",
    "ELL", "ELN", "EML4", "EP300", "EPAS1", "EPCAM", "EPHA3",
    "EPHA7", "EPHB1", "EPS15", "ERBB2", "ERBB3", "ERBB4", "ERC1",
    "ERCC2", "ERCC3", "ERCC4", "ERCC5", "ERG", "ESR1", "ETNK1",
    "ETV1", "ETV4", "ETV5", "ETV6", "EWSR1", "EXT1", "EXT2", "EZH2",
    "EZR", "FAM131B", "FAM135B", "FAM47C", "FANCA", "FANCC", "FANCD2",
    "FANCE", "FANCF", "FANCG", "FAS", "FAT1", "FAT3", "FAT4",
    "FBLN2", "FBXO11", "FBXW7", "FCGR2B", "FCRL4", "FEN1", "FES",
    "FEV", "FGFR1", "FGFR1OP", "FGFR2", "FGFR3", "FGFR4", "FH",
    "FHIT", "FIP1L1", "FLCN", "FLI1", "FLNA", "FLT3", "FLT4",
    "FNBP1", "FOXA1", "FOXL2", "FOXO1", "FOXO3", "FOXO4", "FOXP1",
    "FOXR1", "FSTL3", "FUBP1", "FUS", "GAB1", "GAB2", "GAS7",
    "GATA1", "GATA2", "GATA3", "GID4", "GLI1", "GMPS", "GNA11",
    "GNA13", "GNAQ", "GNAS", "GPC3", "GPC5", "GPHN", "GPS2",
    "GRIN2A", "GRM3", "H3-3A", "H3-3B", "H3C2", "HCFC1", "HDAC1",
    "HEY1", "HIF1A", "HIP1", "HIST1H3B", "HLA-A", "HLF", "HMGA1",
    "HMGA2", "HNF1A", "HNRNPA2B1", "HOOK3", "HOXA11", "HOXA13",
    "HOXA9", "HOXC11", "HOXC13", "HOXD11", "HOXD13", "HRAS", "HSP90AA1",
    "HSP90AB1", "ID3", "IDH1", "IDH2", "IGF1R", "IGF2", "IKBKB",
    "IKZF1", "IL2", "IL21R", "IL6ST", "IL7R", "IRF4", "IRS4",
    "ISX", "ITPKB", "ITK", "JAK1", "JAK2", "JAK3", "JAZF1",
    "JUN", "KAT6A", "KAT6B", "KAT7", "KCNJ5", "KDM5A", "KDM5C",
    "KDM6A", "KDR", "KDSR", "KEAP1", "KEL", "KIF5B", "KLHL6",
    "KLK2", "KMT2A", "KMT2B", "KMT2C", "KMT2D", "KNSTRN", "KRAS",
    "KTN1", "LARP4B", "LASP1", "LCK", "LCP1", "LEF1", "LHFPL6",
    "LIFR", "LMNA", "LMO1", "LMO2", "LPP", "LRIG3", "LRP1B",
    "LSM14A", "LYL1", "LZTR1", "MAF", "MAFB", "MALT1", "MAML2",
    "MAP2K1", "MAP2K2", "MAP2K4", "MAP3K1", "MAP3K13", "MAP3K14",
    "MAPK1", "MAX", "MB21D2", "MDM2", "MDM4", "MDS2", "MECOM",
    "MED12", "MEN1", "MET", "MITF", "MKL1", "MLH1", "MLLT1",
    "MLLT10", "MLLT11", "MLLT3", "MLLT6", "MN1", "MNX1", "MPL",
    "MSH2", "MSH3", "MSH6", "MSI2", "MSN", "MTCP1", "MTOR",
    "MUC1", "MUC16", "MUC4", "MUTYH", "MYB", "MYC", "MYCL",
    "MYCN", "MYD88", "MYH11", "MYH9", "MYO5A", "MYOD1", "NAB2",
    "NACA", "NBN", "NCKIPSD", "NCOA1", "NCOA2", "NCOA4", "NCOR1",
    "NCOR2", "NDRG1", "NF1", "NF2", "NFATC2", "NFE2L2", "NFIB",
    "NFKB2", "NFKBIA", "NIN", "NKX2-1", "NONO", "NOTCH1", "NOTCH2",
    "NPM1", "NR4A3", "NRAS", "NRG1", "NSD1", "NSD2", "NSD3",
    "NT5C2", "NTHL1", "NTRK1", "NTRK2", "NTRK3", "NUP214", "NUP98",
    "NUTM1", "NUTM2A", "NUTM2B", "OLIG2", "OMD", "P2RY8", "PAFAH1B2",
    "PALB2", "PATZ1", "PAX3", "PAX5", "PAX7", "PAX8", "PBRM1",
    "PBX1", "PCBP1", "PCM1", "PDCD1LG2", "PDE4DIP", "PDGFB",
    "PDGFRA", "PDGFRB", "PER1", "PHF6", "PHOX2B", "PICALM", "PIK3CA",
    "PIK3CB", "PIK3R1", "PIK3R2", "PIM1", "PLAG1", "PLCG1", "PML",
    "PMS1", "PMS2", "POLD1", "POLE", "POLG", "POLQ", "POT1",
    "POU2AF1", "POU5F1", "PPARG", "PPFIBP1", "PPM1D", "PPP2R1A",
    "PPP6C", "PRCC", "PRDM1", "PRDM16", "PRDM2", "PREX2", "PRF1",
    "PRKACA", "PRKAR1A", "PRKCB", "PRRX1", "PSIP1", "PTCH1",
    "PTEN", "PTK6", "PTPN11", "PTPN13", "PTPN6", "PTPRB", "PTPRC",
    "PTPRD", "PTPRK", "PTPRT", "QKI", "RABEP1", "RAC1", "RAD21",
    "RAD51B", "RAD51C", "RAD51D", "RAF1", "RALGDS", "RANBP2",
    "RAP1GDS1", "RARA", "RB1", "RBM10", "RBM15", "RECQL4", "REL",
    "RET", "RFWD3", "RGPD3", "RGS7", "RHOA", "RHOH", "RMI2",
    "RNF213", "RNF43", "ROBO2", "ROS1", "RPL10", "RPL22", "RPL5",
    "RPN1", "RSPO2", "RSPO3", "RUNX1", "RUNX1T1", "S1PR2", "SALL4",
    "SBDS", "SDC4", "SDHA", "SDHAF2", "SDHB", "SDHC", "SDHD",
    "SEPTIN5", "SEPTIN6", "SEPTIN9", "SET", "SETBP1", "SETD2",
    "SF3B1", "SFPQ", "SFRP4", "SGK1", "SH2B3", "SH3GL1", "SHTN1",
    "SIX1", "SIX2", "SKI", "SLC34A2", "SLC45A3", "SMAD2", "SMAD3",
    "SMAD4", "SMARCA4", "SMARCB1", "SMARCD1", "SMARCE1", "SMO",
    "SND1", "SNX29", "SOCS1", "SOX2", "SOX9", "SPECC1", "SPEN",
    "SPOP", "SRC", "SRGAP3", "SRSF2", "SRSF3", "SS18", "SS18L1",
    "SSX1", "SSX2", "SSX4", "STAG1", "STAG2", "STAT3", "STAT5B",
    "STAT6", "STIL", "STK11", "STRN", "SUFU", "SUZ12", "SYK",
    "TAF15", "TAL1", "TAL2", "TBL1XR1", "TBX3", "TCEA1", "TCF12",
    "TCF3", "TCF7L2", "TCL1A", "TEC", "TENT5C", "TERT", "TET1",
    "TET2", "TFE3", "TFEB", "TFG", "TFPT", "TFRC", "TGFBR2",
    "THRAP3", "TLX1", "TLX3", "TMEM127", "TMPRSS2", "TNFAIP3",
    "TNFRSF14", "TNFRSF17", "TOP1", "TP53", "TP63", "TPM3", "TPM4",
    "TPR", "TRAF7", "TRIM24", "TRIM27", "TRIM33", "TRIP11", "TRRAP",
    "TSC1", "TSC2", "TSHR", "U2AF1", "UBR5", "USP44", "USP6",
    "USP8", "VAV1", "VHL", "VOPP1", "WAS", "WDCP", "WIF1",
    "WNK2", "WRN", "WT1", "WWTR1", "XPA", "XPC", "XPO1",
    "YWHAE", "ZBTB16", "ZCCHC8", "ZEB1", "ZFHX3", "ZMYM2", "ZMYM3",
    "ZNF331", "ZNF384", "ZNF429", "ZNF479", "ZNF521", "ZNRF3",
    "ZRSR2",
})

REVIEW_STAR_MAP: dict[str, int] = {
    "practice guideline": 4,
    "reviewed by expert panel": 3,
    "criteria provided, multiple submitters, no conflicts": 2,
    "criteria provided, conflicting classifications": 1,
    "criteria provided, single submitter": 1,
    "no assertion criteria provided": 0,
    "no assertion provided": 0,
    "no classification provided": 0,
}


def load_cosmic_genes(path: Path | None = None) -> frozenset[str]:
    """Load COSMIC Cancer Gene Census genes.

    Args:
        path: Optional path to a CSV file with a ``Gene Symbol`` column.
            If ``None``, uses the built-in list.

    Returns:
        Frozenset of gene symbols.
    """
    if path is not None and path.is_file():
        df = pd.read_csv(path)
        col = "Gene Symbol" if "Gene Symbol" in df.columns else df.columns[0]
        genes = frozenset(df[col].astype(str).str.strip())
        logger.info("Loaded %d COSMIC genes from %s", len(genes), path)
        return genes

    logger.info("Using built-in COSMIC Cancer Gene Census (%d genes)", len(COSMIC_CENSUS_GENES))
    return COSMIC_CENSUS_GENES


def _map_review_stars(review_status: pd.Series) -> pd.Series:
    """Map ClinVar review status strings to star counts.

    Args:
        review_status: Series of review status strings.

    Returns:
        Series of integer star counts.
    """
    normalised = review_status.astype(str).str.strip().str.lower()
    return normalised.map(REVIEW_STAR_MAP).fillna(0).astype(int)


def validate_cancer_driver_predictions(
    gene_symbols: np.ndarray | pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    cosmic_genes: frozenset[str] | None = None,
) -> dict[str, Any]:
    """Validate predictions for known cancer driver genes vs non-cancer genes.

    Args:
        gene_symbols: Gene symbol for each sample.
        y_true: True labels of shape ``(n,)``.
        y_pred: Predicted labels of shape ``(n,)``.
        y_prob: Predicted probabilities of shape ``(n, num_classes)``.
        cosmic_genes: Set of COSMIC cancer driver gene symbols.

    Returns:
        Dict with driver and non-driver prediction statistics.
    """
    if cosmic_genes is None:
        cosmic_genes = COSMIC_CENSUS_GENES

    genes = np.asarray(gene_symbols, dtype=str)
    is_driver = np.array([g in cosmic_genes for g in genes])

    result: dict[str, Any] = {
        "n_total": len(genes),
        "n_driver_genes": int(is_driver.sum()),
        "n_non_driver_genes": int((~is_driver).sum()),
    }

    if is_driver.sum() > 0:
        driver_true = y_true[is_driver]
        driver_pred = y_pred[is_driver]
        driver_prob = y_prob[is_driver]

        pathogenic_mask = (driver_true == 0) | (driver_true == 1)
        if pathogenic_mask.sum() > 0:
            driver_pred_pathogenic = (driver_pred[pathogenic_mask] == 0) | (
                driver_pred[pathogenic_mask] == 1
            )
            result["driver_pathogenic_recall"] = float(driver_pred_pathogenic.mean())
        else:
            result["driver_pathogenic_recall"] = float("nan")

        result["driver_accuracy"] = float((driver_pred == driver_true).mean())
        result["driver_mean_confidence"] = float(
            np.max(driver_prob, axis=1).mean()
        )

    if (~is_driver).sum() > 0:
        nondriver_true = y_true[~is_driver]
        nondriver_pred = y_pred[~is_driver]
        nondriver_prob = y_prob[~is_driver]

        benign_mask = (nondriver_true == 2) | (nondriver_true == 3)
        if benign_mask.sum() > 0:
            nondriver_pred_benign = (nondriver_pred[benign_mask] == 2) | (
                nondriver_pred[benign_mask] == 3
            )
            result["nondriver_benign_recall"] = float(nondriver_pred_benign.mean())
        else:
            result["nondriver_benign_recall"] = float("nan")

        result["nondriver_accuracy"] = float((nondriver_pred == nondriver_true).mean())
        result["nondriver_mean_confidence"] = float(
            np.max(nondriver_prob, axis=1).mean()
        )

    return result


def validate_clinvar_confidence(
    review_status: np.ndarray | pd.Series,
    y_prob: np.ndarray,
) -> dict[str, Any]:
    """Check if model confidence correlates with ClinVar review star count.

    Args:
        review_status: ClinVar review status strings for each sample.
        y_prob: Predicted probabilities of shape ``(n, num_classes)``.

    Returns:
        Dict mapping star count to mean model confidence and sample count.
    """
    stars = _map_review_stars(pd.Series(review_status))
    confidences = np.max(y_prob, axis=1)

    result: dict[str, Any] = {"per_star": {}}
    for star_val in sorted(stars.unique()):
        mask = stars == star_val
        result["per_star"][int(star_val)] = {
            "n_samples": int(mask.sum()),
            "mean_confidence": float(confidences[mask].mean()),
            "std_confidence": float(confidences[mask].std()),
        }

    star_values = stars.values.astype(float)
    if len(np.unique(star_values)) > 1:
        correlation = float(np.corrcoef(star_values, confidences)[0, 1])
        result["star_confidence_correlation"] = correlation
    else:
        result["star_confidence_correlation"] = float("nan")

    return result


def gene_level_accuracy(
    gene_symbols: np.ndarray | pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    min_samples: int = 5,
) -> pd.DataFrame:
    """Compute per-gene accuracy.

    Args:
        gene_symbols: Gene symbol for each sample.
        y_true: True labels of shape ``(n,)``.
        y_pred: Predicted labels of shape ``(n,)``.
        min_samples: Minimum samples for a gene to be included.

    Returns:
        DataFrame with gene, accuracy, n_samples, n_correct columns,
        sorted by accuracy ascending.
    """
    df = pd.DataFrame({
        "gene": np.asarray(gene_symbols, dtype=str),
        "y_true": y_true,
        "y_pred": y_pred,
        "correct": (y_true == y_pred).astype(int),
    })

    gene_stats = df.groupby("gene").agg(
        n_samples=("correct", "count"),
        n_correct=("correct", "sum"),
    ).reset_index()

    gene_stats = gene_stats[gene_stats["n_samples"] >= min_samples].copy()
    gene_stats["accuracy"] = gene_stats["n_correct"] / gene_stats["n_samples"]
    gene_stats = gene_stats.sort_values("accuracy", ascending=True).reset_index(drop=True)

    return gene_stats


def cancer_driver_classification_report(
    gene_symbols: np.ndarray | pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cosmic_genes: frozenset[str] | None = None,
) -> dict[str, float]:
    """Compute precision/recall/F1 for cancer driver mutation predictions.

    Considers labels 0 (Pathogenic) and 1 (Likely Pathogenic) as positive
    for variants in known cancer driver genes.

    Args:
        gene_symbols: Gene symbol for each sample.
        y_true: True labels of shape ``(n,)``.
        y_pred: Predicted labels of shape ``(n,)``.
        cosmic_genes: Set of COSMIC cancer driver gene symbols.

    Returns:
        Dict with precision, recall, F1 for driver mutations.
    """
    if cosmic_genes is None:
        cosmic_genes = COSMIC_CENSUS_GENES

    genes = np.asarray(gene_symbols, dtype=str)
    is_driver = np.array([g in cosmic_genes for g in genes])

    if is_driver.sum() == 0:
        return {
            "driver_precision": float("nan"),
            "driver_recall": float("nan"),
            "driver_f1": float("nan"),
        }

    driver_true = y_true[is_driver]
    driver_pred = y_pred[is_driver]

    binary_true = ((driver_true == 0) | (driver_true == 1)).astype(int)
    binary_pred = ((driver_pred == 0) | (driver_pred == 1)).astype(int)

    return {
        "driver_precision": float(precision_score(binary_true, binary_pred, zero_division=0)),
        "driver_recall": float(recall_score(binary_true, binary_pred, zero_division=0)),
        "driver_f1": float(f1_score(binary_true, binary_pred, zero_division=0)),
    }


def run_biological_validation(
    gene_symbols: np.ndarray | pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    review_status: np.ndarray | pd.Series | None = None,
    cosmic_path: Path | None = None,
) -> dict[str, Any]:
    """Run all biological validation analyses.

    Args:
        gene_symbols: Gene symbol for each sample.
        y_true: True labels of shape ``(n,)``.
        y_pred: Predicted labels of shape ``(n,)``.
        y_prob: Predicted probabilities of shape ``(n, num_classes)``.
        review_status: Optional ClinVar review status strings.
        cosmic_path: Optional path to COSMIC gene census CSV.

    Returns:
        Combined dict of all validation results.
    """
    cosmic_genes = load_cosmic_genes(cosmic_path)

    results: dict[str, Any] = {}

    results["driver_validation"] = validate_cancer_driver_predictions(
        gene_symbols, y_true, y_pred, y_prob, cosmic_genes,
    )

    results["driver_classification"] = cancer_driver_classification_report(
        gene_symbols, y_true, y_pred, cosmic_genes,
    )

    if review_status is not None:
        results["clinvar_confidence"] = validate_clinvar_confidence(
            review_status, y_prob,
        )

    gene_acc = gene_level_accuracy(gene_symbols, y_true, y_pred, min_samples=5)
    results["gene_level_accuracy"] = {
        "n_genes_evaluated": len(gene_acc),
        "mean_gene_accuracy": float(gene_acc["accuracy"].mean()) if len(gene_acc) > 0 else 0.0,
        "worst_10_genes": gene_acc.head(10).to_dict("records") if len(gene_acc) > 0 else [],
        "best_10_genes": gene_acc.tail(10).to_dict("records") if len(gene_acc) > 0 else [],
    }

    logger.info("Biological validation complete:")
    dv = results["driver_validation"]
    logger.info(
        "  Driver genes: %d samples, accuracy=%.4f, pathogenic recall=%.4f",
        dv.get("n_driver_genes", 0),
        dv.get("driver_accuracy", 0),
        dv.get("driver_pathogenic_recall", 0),
    )
    logger.info(
        "  Non-driver genes: %d samples, accuracy=%.4f, benign recall=%.4f",
        dv.get("n_non_driver_genes", 0),
        dv.get("nondriver_accuracy", 0),
        dv.get("nondriver_benign_recall", 0),
    )
    dc = results["driver_classification"]
    logger.info(
        "  Driver P/R/F1: %.4f / %.4f / %.4f",
        dc.get("driver_precision", 0),
        dc.get("driver_recall", 0),
        dc.get("driver_f1", 0),
    )

    return results
