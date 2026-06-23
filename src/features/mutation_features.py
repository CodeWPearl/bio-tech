"""Mutation-level feature extraction for pathogenicity prediction.

Transforms raw somatic mutation calls (gene symbol, variant classification,
protein change, chromosome, position, alleles) into a fixed-width numeric
feature vector suitable for downstream encoders.

Feature groups:

* **Variant type one-hot** — 8 categories covering the major functional classes.
* **Amino acid change properties** — biochemical distance metrics (Grantham,
  BLOSUM62) and physicochemical deltas (hydrophobicity, charge, molecular
  weight) parsed from the protein-change string.
* **Gene-level features** — COSMIC Cancer Gene Census membership, per-gene
  mutation frequency (learned from training data), and gene coding-sequence
  length.
* **Positional features** — chromosome one-hot (24 values) and normalised
  position within the gene.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Amino acid property tables
# ---------------------------------------------------------------------------

#: Kyte-Doolittle hydrophobicity index for the 20 standard amino acids.
AA_HYDROPHOBICITY: dict[str, float] = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "E": -3.5, "Q": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

#: Net charge at physiological pH.
AA_CHARGE: dict[str, float] = {
    "A": 0, "R": 1, "N": 0, "D": -1, "C": 0,
    "E": -1, "Q": 0, "G": 0, "H": 0.5, "I": 0,
    "L": 0, "K": 1, "M": 0, "F": 0, "P": 0,
    "S": 0, "T": 0, "W": 0, "Y": 0, "V": 0,
}

#: Molecular weight (Da) of each amino acid residue.
AA_WEIGHT: dict[str, float] = {
    "A": 89.1, "R": 174.2, "N": 132.1, "D": 133.1, "C": 121.2,
    "E": 147.1, "Q": 146.1, "G": 75.0, "H": 155.2, "I": 131.2,
    "L": 131.2, "K": 146.2, "M": 149.2, "F": 165.2, "P": 115.1,
    "S": 105.1, "T": 119.1, "W": 204.2, "Y": 181.2, "V": 117.1,
}

#: Grantham distance matrix (symmetric).  Only the upper-triangle pairs are
#: stored; lookups check both orderings.
GRANTHAM: dict[tuple[str, str], int] = {
    ("A", "R"): 112, ("A", "N"): 111, ("A", "D"): 126, ("A", "C"): 195,
    ("A", "E"): 107, ("A", "Q"): 91, ("A", "G"): 60, ("A", "H"): 86,
    ("A", "I"): 94, ("A", "L"): 96, ("A", "K"): 106, ("A", "M"): 84,
    ("A", "F"): 113, ("A", "P"): 27, ("A", "S"): 99, ("A", "T"): 58,
    ("A", "W"): 148, ("A", "Y"): 112, ("A", "V"): 64,
    ("R", "N"): 86, ("R", "D"): 96, ("R", "C"): 180, ("R", "E"): 54,
    ("R", "Q"): 43, ("R", "G"): 125, ("R", "H"): 29, ("R", "I"): 97,
    ("R", "L"): 102, ("R", "K"): 26, ("R", "M"): 91, ("R", "F"): 97,
    ("R", "P"): 103, ("R", "S"): 110, ("R", "T"): 71, ("R", "W"): 101,
    ("R", "Y"): 77, ("R", "V"): 96,
    ("N", "D"): 23, ("N", "C"): 139, ("N", "E"): 42, ("N", "Q"): 46,
    ("N", "G"): 80, ("N", "H"): 68, ("N", "I"): 149, ("N", "L"): 153,
    ("N", "K"): 94, ("N", "M"): 142, ("N", "F"): 158, ("N", "P"): 91,
    ("N", "S"): 46, ("N", "T"): 65, ("N", "W"): 174, ("N", "Y"): 143,
    ("N", "V"): 133,
    ("D", "C"): 154, ("D", "E"): 45, ("D", "Q"): 61, ("D", "G"): 94,
    ("D", "H"): 81, ("D", "I"): 168, ("D", "L"): 172, ("D", "K"): 101,
    ("D", "M"): 160, ("D", "F"): 177, ("D", "P"): 108, ("D", "S"): 65,
    ("D", "T"): 85, ("D", "W"): 181, ("D", "Y"): 160, ("D", "V"): 152,
    ("C", "E"): 170, ("C", "Q"): 154, ("C", "G"): 159, ("C", "H"): 174,
    ("C", "I"): 198, ("C", "L"): 198, ("C", "K"): 202, ("C", "M"): 196,
    ("C", "F"): 205, ("C", "P"): 169, ("C", "S"): 112, ("C", "T"): 149,
    ("C", "W"): 215, ("C", "Y"): 194, ("C", "V"): 192,
    ("E", "Q"): 29, ("E", "G"): 98, ("E", "H"): 40, ("E", "I"): 134,
    ("E", "L"): 138, ("E", "K"): 56, ("E", "M"): 126, ("E", "F"): 140,
    ("E", "P"): 93, ("E", "S"): 80, ("E", "T"): 65, ("E", "W"): 152,
    ("E", "Y"): 122, ("E", "V"): 121,
    ("Q", "G"): 87, ("Q", "H"): 24, ("Q", "I"): 109, ("Q", "L"): 113,
    ("Q", "K"): 53, ("Q", "M"): 101, ("Q", "F"): 116, ("Q", "P"): 76,
    ("Q", "S"): 68, ("Q", "T"): 42, ("Q", "W"): 130, ("Q", "Y"): 99,
    ("Q", "V"): 96,
    ("G", "H"): 98, ("G", "I"): 135, ("G", "L"): 138, ("G", "K"): 127,
    ("G", "M"): 127, ("G", "F"): 153, ("G", "P"): 42, ("G", "S"): 56,
    ("G", "T"): 59, ("G", "W"): 184, ("G", "Y"): 147, ("G", "V"): 109,
    ("H", "I"): 94, ("H", "L"): 99, ("H", "K"): 32, ("H", "M"): 87,
    ("H", "F"): 100, ("H", "P"): 77, ("H", "S"): 89, ("H", "T"): 47,
    ("H", "W"): 115, ("H", "Y"): 83, ("H", "V"): 84,
    ("I", "L"): 5, ("I", "K"): 102, ("I", "M"): 10, ("I", "F"): 21,
    ("I", "P"): 95, ("I", "S"): 142, ("I", "T"): 89, ("I", "W"): 61,
    ("I", "Y"): 33, ("I", "V"): 29,
    ("L", "K"): 107, ("L", "M"): 15, ("L", "F"): 22, ("L", "P"): 98,
    ("L", "S"): 145, ("L", "T"): 92, ("L", "W"): 61, ("L", "Y"): 36,
    ("L", "V"): 32,
    ("K", "M"): 95, ("K", "F"): 102, ("K", "P"): 103, ("K", "S"): 121,
    ("K", "T"): 78, ("K", "W"): 110, ("K", "Y"): 85, ("K", "V"): 97,
    ("M", "F"): 28, ("M", "P"): 87, ("M", "S"): 135, ("M", "T"): 81,
    ("M", "W"): 67, ("M", "Y"): 36, ("M", "V"): 21,
    ("F", "P"): 114, ("F", "S"): 155, ("F", "T"): 103, ("F", "W"): 40,
    ("F", "Y"): 22, ("F", "V"): 50,
    ("P", "S"): 74, ("P", "T"): 38, ("P", "W"): 147, ("P", "Y"): 110,
    ("P", "V"): 68,
    ("S", "T"): 58, ("S", "W"): 177, ("S", "Y"): 144, ("S", "V"): 124,
    ("T", "W"): 128, ("T", "Y"): 92, ("T", "V"): 69,
    ("W", "Y"): 37, ("W", "V"): 88,
    ("Y", "V"): 55,
}

#: BLOSUM62 substitution scores (symmetric, including diagonal).
BLOSUM62: dict[tuple[str, str], int] = {
    ("A", "A"): 4, ("A", "R"): -1, ("A", "N"): -2, ("A", "D"): -2,
    ("A", "C"): 0, ("A", "E"): -1, ("A", "Q"): -1, ("A", "G"): 0,
    ("A", "H"): -2, ("A", "I"): -1, ("A", "L"): -1, ("A", "K"): -1,
    ("A", "M"): -1, ("A", "F"): -2, ("A", "P"): -1, ("A", "S"): 1,
    ("A", "T"): 0, ("A", "W"): -3, ("A", "Y"): -2, ("A", "V"): 0,
    ("R", "R"): 5, ("R", "N"): 0, ("R", "D"): -2, ("R", "C"): -3,
    ("R", "E"): 0, ("R", "Q"): 1, ("R", "G"): -2, ("R", "H"): 0,
    ("R", "I"): -3, ("R", "L"): -2, ("R", "K"): 2, ("R", "M"): -1,
    ("R", "F"): -3, ("R", "P"): -2, ("R", "S"): -1, ("R", "T"): -1,
    ("R", "W"): -3, ("R", "Y"): -2, ("R", "V"): -3,
    ("N", "N"): 6, ("N", "D"): 1, ("N", "C"): -3, ("N", "E"): 0,
    ("N", "Q"): 0, ("N", "G"): 0, ("N", "H"): 1, ("N", "I"): -3,
    ("N", "L"): -3, ("N", "K"): 0, ("N", "M"): -2, ("N", "F"): -3,
    ("N", "P"): -2, ("N", "S"): 1, ("N", "T"): 0, ("N", "W"): -4,
    ("N", "Y"): -2, ("N", "V"): -3,
    ("D", "D"): 6, ("D", "C"): -3, ("D", "E"): 2, ("D", "Q"): 0,
    ("D", "G"): -1, ("D", "H"): -1, ("D", "I"): -3, ("D", "L"): -4,
    ("D", "K"): -1, ("D", "M"): -3, ("D", "F"): -3, ("D", "P"): -1,
    ("D", "S"): 0, ("D", "T"): -1, ("D", "W"): -4, ("D", "Y"): -3,
    ("D", "V"): -3,
    ("C", "C"): 9, ("C", "E"): -4, ("C", "Q"): -3, ("C", "G"): -3,
    ("C", "H"): -3, ("C", "I"): -1, ("C", "L"): -1, ("C", "K"): -3,
    ("C", "M"): -1, ("C", "F"): -2, ("C", "P"): -3, ("C", "S"): -1,
    ("C", "T"): -1, ("C", "W"): -2, ("C", "Y"): -2, ("C", "V"): -1,
    ("E", "E"): 5, ("E", "Q"): 2, ("E", "G"): -2, ("E", "H"): 0,
    ("E", "I"): -3, ("E", "L"): -3, ("E", "K"): 1, ("E", "M"): -2,
    ("E", "F"): -3, ("E", "P"): -1, ("E", "S"): 0, ("E", "T"): -1,
    ("E", "W"): -3, ("E", "Y"): -2, ("E", "V"): -2,
    ("Q", "Q"): 5, ("Q", "G"): -2, ("Q", "H"): 0, ("Q", "I"): -3,
    ("Q", "L"): -2, ("Q", "K"): 1, ("Q", "M"): 0, ("Q", "F"): -3,
    ("Q", "P"): -1, ("Q", "S"): 0, ("Q", "T"): -1, ("Q", "W"): -2,
    ("Q", "Y"): -1, ("Q", "V"): -2,
    ("G", "G"): 6, ("G", "H"): -2, ("G", "I"): -4, ("G", "L"): -4,
    ("G", "K"): -2, ("G", "M"): -3, ("G", "F"): -3, ("G", "P"): -2,
    ("G", "S"): 0, ("G", "T"): -2, ("G", "W"): -2, ("G", "Y"): -3,
    ("G", "V"): -3,
    ("H", "H"): 8, ("H", "I"): -3, ("H", "L"): -3, ("H", "K"): -1,
    ("H", "M"): -2, ("H", "F"): -1, ("H", "P"): -2, ("H", "S"): -1,
    ("H", "T"): -2, ("H", "W"): -2, ("H", "Y"): 2, ("H", "V"): -3,
    ("I", "I"): 4, ("I", "L"): 2, ("I", "K"): -3, ("I", "M"): 1,
    ("I", "F"): 0, ("I", "P"): -3, ("I", "S"): -2, ("I", "T"): -1,
    ("I", "W"): -3, ("I", "Y"): -1, ("I", "V"): 3,
    ("L", "L"): 4, ("L", "K"): -2, ("L", "M"): 2, ("L", "F"): 0,
    ("L", "P"): -3, ("L", "S"): -2, ("L", "T"): -1, ("L", "W"): -2,
    ("L", "Y"): -1, ("L", "V"): 1,
    ("K", "K"): 5, ("K", "M"): -1, ("K", "F"): -3, ("K", "P"): -1,
    ("K", "S"): 0, ("K", "T"): -1, ("K", "W"): -3, ("K", "Y"): -2,
    ("K", "V"): -2,
    ("M", "M"): 5, ("M", "F"): 0, ("M", "P"): -2, ("M", "S"): -1,
    ("M", "T"): -1, ("M", "W"): -1, ("M", "Y"): -1, ("M", "V"): 1,
    ("F", "F"): 6, ("F", "P"): -4, ("F", "S"): -2, ("F", "T"): -2,
    ("F", "W"): 1, ("F", "Y"): 3, ("F", "V"): -1,
    ("P", "P"): 7, ("P", "S"): -1, ("P", "T"): -1, ("P", "W"): -4,
    ("P", "Y"): -3, ("P", "V"): -2,
    ("S", "S"): 4, ("S", "T"): 1, ("S", "W"): -3, ("S", "Y"): -2,
    ("S", "V"): -2,
    ("T", "T"): 5, ("T", "W"): -2, ("T", "Y"): -2, ("T", "V"): 0,
    ("W", "W"): 11, ("W", "Y"): 2, ("W", "V"): -3,
    ("Y", "Y"): 7, ("Y", "V"): -1,
    ("V", "V"): 4,
}

# ---------------------------------------------------------------------------
# COSMIC Cancer Gene Census — Tier 1 + Tier 2 driver genes (curated subset)
# ---------------------------------------------------------------------------

COSMIC_CENSUS_GENES: frozenset[str] = frozenset({
    "ABL1", "ABL2", "ACKR3", "ACSL3", "ACSL6", "ACVR1", "ACVR2A",
    "AFF1", "AFF3", "AFF4", "AJUBA", "AKT1", "AKT2", "AKT3",
    "ALDH2", "ALK", "AMER1", "ANK1", "APC", "APOBEC3B", "AR",
    "ARAF", "ARHGAP26", "ARHGAP35", "ARHGEF12", "ARID1A", "ARID1B",
    "ARID2", "ARNT", "ASXL1", "ASXL2", "ATF1", "ATIC", "ATM",
    "ATP1A1", "ATP2B3", "ATR", "ATRX", "AXIN1", "AXIN2", "B2M",
    "BAP1", "BARD1", "BCL10", "BCL11A", "BCL11B", "BCL2", "BCL2L12",
    "BCL3", "BCL6", "BCL7A", "BCL9", "BCL9L", "BCOR", "BCORL1",
    "BCR", "BIRC3", "BIRC6", "BLM", "BMPR1A", "BRAF", "BRCA1",
    "BRCA2", "BRD3", "BRD4", "BRIP1", "BTG1", "BTK", "BUB1B",
    "CACNA1D", "CALR", "CAMTA1", "CARD11", "CARS1", "CASP8",
    "CBFA2T3", "CBFB", "CBL", "CBLB", "CBLC", "CCDC6", "CCNB1IP1",
    "CCND1", "CCND2", "CCND3", "CCNE1", "CD274", "CD28", "CD74",
    "CD79A", "CD79B", "CDC73", "CDH1", "CDH10", "CDH11", "CDH17",
    "CDK12", "CDK4", "CDK6", "CDKN1A", "CDKN1B", "CDKN2A", "CDKN2B",
    "CDKN2C", "CDX2", "CEBPA", "CHD2", "CHD4", "CHEK2", "CHIC2",
    "CHN1", "CIC", "CIITA", "CLIP1", "CLP1", "CLTC", "CLTCL1",
    "CNBP", "CNOT3", "CNTRL", "COL1A1", "COL2A1", "COL3A1",
    "CREB1", "CREB3L1", "CREB3L2", "CREBBP", "CRLF2", "CRTC1",
    "CRTC3", "CSF1R", "CSF3R", "CTCF", "CTLA4", "CTNNA1",
    "CTNNB1", "CUX1", "CXCR4", "CYLD", "CYP2C8", "DAXX",
    "DCAF12L2", "DCC", "DCTN1", "DDB2", "DDIT3", "DDR2", "DDX10",
    "DDX3X", "DDX41", "DDX5", "DDX6", "DEK", "DGCR8", "DICER1",
    "DIS3", "DKC1", "DNAJB1", "DNM2", "DNMT3A", "DNMT3B",
    "DOT1L", "DROSHA", "DTX1", "DUSP22", "EBF1", "ECT2L",
    "EED", "EGFR", "EIF1AX", "EIF3E", "EIF4A2", "ELF3", "ELF4",
    "ELK4", "ELL", "ELN", "EML4", "EP300", "EPAS1", "EPCAM",
    "EPHA3", "EPHA7", "EPHB1", "EPS15", "ERBB2", "ERBB3", "ERBB4",
    "ERC1", "ERCC2", "ERCC3", "ERCC4", "ERCC5", "ERG", "ESR1",
    "ETNK1", "ETV1", "ETV4", "ETV5", "ETV6", "EWSR1", "EXT1",
    "EXT2", "EZH2", "EZR", "FAM131B", "FAM135A", "FAM46C",
    "FANCA", "FANCC", "FANCD2", "FANCE", "FANCF", "FANCG",
    "FANCL", "FAS", "FAT1", "FAT3", "FAT4", "FBLN2", "FBXO11",
    "FBXW7", "FCGR2B", "FCRL4", "FEN1", "FES", "FEV", "FGFR1",
    "FGFR1OP", "FGFR2", "FGFR3", "FGFR4", "FH", "FHIT",
    "FIP1L1", "FKBP9", "FLCN", "FLI1", "FLNA", "FLT1", "FLT3",
    "FLT4", "FNBP1", "FOXA1", "FOXL2", "FOXO1", "FOXO3",
    "FOXO4", "FOXP1", "FOXR1", "FSTL3", "FUBP1", "FUS",
    "GAB1", "GAB2", "GAS7", "GATA1", "GATA2", "GATA3",
    "GLI1", "GMPS", "GNA11", "GNA13", "GNAQ", "GNAS",
    "GPC3", "GPC5", "GPHN", "GPS2", "GRIN2A", "GRM3",
    "H3-3A", "H3-3B", "H3C2", "HCAR1", "HGF", "HIST1H3B",
    "HIF1A", "HIP1", "HMGA1", "HMGA2", "HNF1A", "HNRNPA2B1",
    "HOOK3", "HOXA11", "HOXA13", "HOXA9", "HOXC11", "HOXC13",
    "HOXD11", "HOXD13", "HRAS", "HSP90AA1", "HSP90AB1",
    "ID3", "IDH1", "IDH2", "IGF1R", "IGF2", "IGF2BP2",
    "IKBKB", "IKZF1", "IL2", "IL21R", "IL6ST", "IL7R",
    "INHBA", "IRF1", "IRF4", "IRS4", "ISX", "ITPKB",
    "JAK1", "JAK2", "JAK3", "JAZF1", "JUN", "KAT6A", "KAT6B",
    "KAT7", "KCNJ5", "KDM5A", "KDM5C", "KDM6A", "KDR",
    "KDSR", "KEAP1", "KEL", "KIF5B", "KIT", "KLF4", "KLF6",
    "KLHL6", "KMT2A", "KMT2B", "KMT2C", "KMT2D", "KNSTRN",
    "KRAS", "KTN1", "LARP4B", "LASP1", "LATS1", "LATS2",
    "LCK", "LEF1", "LMNA", "LMO1", "LMO2", "LPP", "LRIG3",
    "LRP1B", "LSM14A", "LYL1", "LZTR1",
    "MAF", "MAFB", "MALT1", "MAML2", "MAP2K1", "MAP2K2",
    "MAP2K4", "MAP3K1", "MAP3K13", "MAP3K14", "MAPK1", "MAX",
    "MB21D2", "MDM2", "MDM4", "MDS2", "MECOM", "MED12",
    "MEN1", "MET", "MITF", "MKL1", "MLH1", "MLLT1",
    "MLLT10", "MLLT3", "MLLT6", "MN1", "MNX1", "MPL",
    "MSH2", "MSH3", "MSH6", "MSI2", "MSN", "MTCP1",
    "MTOR", "MUC1", "MUC16", "MUC4", "MUTYH", "MYB",
    "MYC", "MYCL", "MYCN", "MYD88", "MYH11", "MYH9",
    "MYO5A", "MYOD1", "N4BP2", "NAB2", "NACA", "NBN",
    "NCOA1", "NCOA2", "NCOA4", "NCOR1", "NCOR2", "NDRG1",
    "NF1", "NF2", "NFATC2", "NFE2L2", "NFIB", "NFKB2",
    "NFKBIE", "NIN", "NKX2-1", "NONO", "NOTCH1", "NOTCH2",
    "NPM1", "NR4A3", "NRAS", "NRG1", "NSD1", "NSD2", "NSD3",
    "NT5C2", "NTHL1", "NTRK1", "NTRK2", "NTRK3", "NUP214",
    "NUP93", "NUP98", "NUTM1", "NUTM2A",
    "OLIG2", "OMD", "P2RY8", "PABPC1", "PAFAH1B2",
    "PALB2", "PARK2", "PATZ1", "PAX3", "PAX5", "PAX7",
    "PAX8", "PBRM1", "PBX1", "PCBP1", "PCM1", "PDCD1",
    "PDCD1LG2", "PDE4DIP", "PDGFB", "PDGFRA", "PDGFRB",
    "PER1", "PHF6", "PHOX2B", "PICALM", "PIK3CA", "PIK3CB",
    "PIK3R1", "PIK3R2", "PIM1", "PLCG1", "PML", "PMS1",
    "PMS2", "POLD1", "POLE", "POLG", "POLQ", "POT1",
    "POU2AF1", "POU5F1", "PPARG", "PPFIBP1", "PPM1D",
    "PPP2R1A", "PPP6C", "PRCC", "PRDM1", "PRDM16", "PRDM2",
    "PREX2", "PRF1", "PRKACA", "PRKAR1A", "PRKCB", "PRKCI",
    "PRKDC", "PRPF40B", "PRSS8", "PSIP1", "PTCH1", "PTEN",
    "PTK6", "PTPN11", "PTPN13", "PTPN6", "PTPRB", "PTPRC",
    "PTPRD", "PTPRK", "PTPRT",
    "QKI", "RABEP1", "RAC1", "RAD21", "RAD51B", "RAD51C",
    "RAD51D", "RAF1", "RALGDS", "RANBP2", "RAP1GDS1",
    "RARA", "RB1", "RBM10", "RBM15", "RECQL4", "REL",
    "RET", "RFWD3", "RGPD3", "RHOA", "RHOB", "RIT1",
    "RMI2", "RNF213", "RNF43", "ROBO2", "ROS1", "RPL10",
    "RPL22", "RPL5", "RPN1", "RSPO2", "RSPO3", "RUNX1",
    "RUNX1T1",
    "S1PR2", "SALL4", "SBDS", "SDC4", "SDHA", "SDHAF2",
    "SDHB", "SDHC", "SDHD", "SETBP1", "SETD1B", "SETD2",
    "SF3B1", "SFPQ", "SFRP4", "SGK1", "SH2B3", "SH3GL1",
    "SHTN1", "SIRPA", "SIX1", "SIX2", "SKI", "SLC34A2",
    "SLC45A3", "SMAD2", "SMAD3", "SMAD4", "SMARCA4",
    "SMARCB1", "SMARCD1", "SMARCE1", "SMC1A", "SMC3",
    "SMO", "SND1", "SOCS1", "SOX2", "SOX9", "SOX21",
    "SPECC1", "SPEN", "SPOP", "SRC", "SRGAP3", "SRSF2",
    "SRSF3", "SS18", "SS18L1", "SSX1", "SSX2", "SSX4",
    "STAG1", "STAG2", "STAT3", "STAT5A", "STAT5B", "STAT6",
    "STIL", "STK11", "STK19", "STRN", "SUFU", "SUZ12",
    "SYK",
    "TAF15", "TAL1", "TAL2", "TBL1XR1", "TBX3", "TCEA1",
    "TCF12", "TCF3", "TCF7L2", "TCL1A", "TERT", "TET1",
    "TET2", "TFE3", "TFEB", "TFG", "TFPT", "TFRC",
    "TGFBR2", "THRAP3", "TLX1", "TLX3", "TMEM127",
    "TMPRSS2", "TNFAIP3", "TNFRSF14", "TNFRSF17",
    "TOP1", "TP53", "TP53BP1", "TP63", "TPM3", "TPM4",
    "TPR", "TRAF3", "TRAF7", "TRIM24", "TRIM27", "TRIM33",
    "TRIP11", "TRRAP", "TSC1", "TSC2", "TSHR", "U2AF1",
    "UBR5", "USP44", "USP6", "USP8",
    "VAV1", "VHL", "VKORC1", "VPREB1",
    "WAS", "WDCP", "WIF1", "WISP3", "WNK2", "WRN", "WT1",
    "WWTR1",
    "XPA", "XPC", "XPO1",
    "YWHAE",
    "ZBTB16", "ZBTB2", "ZCCHC8", "ZEB1", "ZFHX3", "ZMYM2",
    "ZMYM3", "ZNF331", "ZNF384", "ZNF429", "ZNF471", "ZNF521",
    "ZNRF3", "ZRSR2",
})

# ---------------------------------------------------------------------------
# Gene coding-sequence lengths (bp).  A representative set of cancer-relevant
# genes; unknown genes fall back to the genome-wide median (~1340 bp).
# ---------------------------------------------------------------------------

DEFAULT_GENE_LENGTH: int = 1340

GENE_LENGTHS: dict[str, int] = {
    "TP53": 1182, "EGFR": 3633, "BRAF": 2301, "KRAS": 567,
    "NRAS": 570, "HRAS": 570, "PIK3CA": 3207, "PTEN": 1212,
    "APC": 8532, "BRCA1": 5592, "BRCA2": 10257, "ATM": 9168,
    "RB1": 2787, "NF1": 8457, "NF2": 1785, "VHL": 639,
    "MET": 4152, "ALK": 4860, "RET": 3345, "KIT": 2928,
    "PDGFRA": 3252, "FGFR1": 2469, "FGFR2": 2466, "FGFR3": 2322,
    "ERBB2": 3768, "ERBB3": 4089, "CDH1": 2649, "SMAD4": 1659,
    "STK11": 1302, "CDKN2A": 471, "CDK4": 912, "CDK6": 981,
    "MYC": 1320, "MYCN": 1362, "IDH1": 1245, "IDH2": 1362,
    "CTNNB1": 2346, "NOTCH1": 7668, "FBXW7": 2169, "ARID1A": 6858,
    "KMT2A": 11919, "KMT2C": 14739, "KMT2D": 16614,
    "DNMT3A": 2739, "TET2": 6063, "WT1": 1491, "EZH2": 2256,
    "SETD2": 7587, "BAP1": 2184, "SMARCA4": 4782, "ATRX": 7452,
    "TTN": 100386, "MTOR": 7650, "JAK2": 3396, "ABL1": 3507,
    "NPM1": 882, "FLT3": 2979, "PTPN11": 1782, "CBL": 2697,
    "GATA3": 1335, "FOXA1": 1401, "ESR1": 1785, "AR": 2763,
    "MAP2K1": 1191, "MAP3K1": 4617, "SF3B1": 3900, "U2AF1": 747,
    "SRSF2": 690, "SPOP": 1143, "NFE2L2": 1803, "KEAP1": 1875,
    "POLE": 6945, "POLD1": 3405, "MSH2": 2805, "MSH6": 4083,
    "MLH1": 2271, "PMS2": 2589,
}

# ---------------------------------------------------------------------------
# Variant classification mapping
# ---------------------------------------------------------------------------

VARIANT_CATEGORIES: tuple[str, ...] = (
    "Missense_Mutation",
    "Nonsense_Mutation",
    "Frame_Shift_Del",
    "Frame_Shift_Ins",
    "Splice_Site",
    "In_Frame_Del",
    "In_Frame_Ins",
    "Silent",
)

#: Aliases that map alternative cBioPortal ``mutation_type`` strings to the
#: canonical variant category used for one-hot encoding.
VARIANT_ALIASES: dict[str, str] = {
    "missense_mutation": "Missense_Mutation",
    "missense": "Missense_Mutation",
    "nonsense_mutation": "Nonsense_Mutation",
    "nonsense": "Nonsense_Mutation",
    "frame_shift_del": "Frame_Shift_Del",
    "frameshift_del": "Frame_Shift_Del",
    "frameshift deletion": "Frame_Shift_Del",
    "frame_shift_ins": "Frame_Shift_Ins",
    "frameshift_ins": "Frame_Shift_Ins",
    "frameshift insertion": "Frame_Shift_Ins",
    "splice_site": "Splice_Site",
    "splice site": "Splice_Site",
    "in_frame_del": "In_Frame_Del",
    "in_frame_ins": "In_Frame_Ins",
    "silent": "Silent",
}

CHROMOSOMES: tuple[str, ...] = (
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
    "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
    "21", "22", "X", "Y",
)

_PROTEIN_CHANGE_RE = re.compile(r"^p\.([A-Z])(\d+)([A-Z])$")
_PROTEIN_CHANGE_LONG_RE = re.compile(
    r"^p\.([A-Za-z]{3})(\d+)([A-Za-z]{3})$"
)

#: Three-letter to one-letter amino acid codes.
AA3_TO_1: dict[str, str] = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C",
    "Glu": "E", "Gln": "Q", "Gly": "G", "His": "H", "Ile": "I",
    "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P",
    "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
}


def _lookup_symmetric(
    table: dict[tuple[str, str], int | float], a: str, b: str
) -> float:
    """Look up a value in a symmetric pair-keyed table."""
    val = table.get((a, b))
    if val is not None:
        return float(val)
    val = table.get((b, a))
    return float(val) if val is not None else 0.0


def _parse_protein_change(text: object) -> tuple[str, str] | None:
    """Extract (ref_aa, alt_aa) single-letter codes from a protein change.

    Handles formats like ``p.R175H``, ``R175H``, ``p.Arg175His``.
    Returns ``None`` when the string is missing or unparseable.
    """
    if not isinstance(text, str) or not text:
        return None
    raw = text.strip()
    if raw.startswith("p."):
        raw = raw[2:]

    m = re.match(r"^([A-Z])(\d+)([A-Z])$", raw)
    if m:
        return m.group(1), m.group(3)

    m = re.match(r"^([A-Za-z]{3})(\d+)([A-Za-z]{3})$", raw)
    if m:
        ref = AA3_TO_1.get(m.group(1).capitalize())
        alt = AA3_TO_1.get(m.group(3).capitalize())
        if ref and alt:
            return ref, alt
    return None


class MutationFeatureExtractor:
    """Extract numeric features from somatic mutation calls.

    Follows the sklearn ``fit`` / ``transform`` convention.  ``fit`` learns
    gene-level statistics (mutation frequency) from the training split;
    ``transform`` produces a fixed-width numpy array.

    Attributes:
        feature_names: Names of the output columns (set after ``fit``).
    """

    def __init__(self) -> None:
        self.feature_names: list[str] = []
        self._fitted: bool = False
        self._gene_freq: dict[str, float] = {}
        self._max_freq: float = 1.0

    def fit(self, df: pd.DataFrame) -> MutationFeatureExtractor:
        """Learn gene-level statistics from training data.

        Args:
            df: Training split of the merged dataset.

        Returns:
            ``self`` for method chaining.
        """
        self._build_feature_names()
        if "gene_symbol" in df.columns:
            counts = df["gene_symbol"].value_counts()
            total = len(df)
            self._gene_freq = (counts / total).to_dict() if total else {}
            self._max_freq = max(self._gene_freq.values(), default=1.0)
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Convert mutation rows to a numeric feature matrix.

        Args:
            df: Merged dataset (or a split thereof).

        Returns:
            Array of shape ``(len(df), len(self.feature_names))``.

        Raises:
            RuntimeError: If :meth:`fit` has not been called.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before transform()")

        n = len(df)
        n_features = len(self.feature_names)
        out = np.zeros((n, n_features), dtype=np.float64)

        if n == 0:
            return out

        offset = 0
        offset = self._encode_variant_type(df, out, offset)
        offset = self._encode_aa_properties(df, out, offset)
        offset = self._encode_gene_features(df, out, offset)
        self._encode_positional(df, out, offset)

        return out

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Convenience: ``fit`` then ``transform``."""
        return self.fit(df).transform(df)

    # --- Feature-name registry -----------------------------------------------

    def _build_feature_names(self) -> None:
        names: list[str] = []
        for cat in VARIANT_CATEGORIES:
            names.append(f"vtype_{cat}")
        names.append("vtype_Other")
        for metric in (
            "grantham", "blosum62", "hydro_delta",
            "charge_delta", "size_delta",
        ):
            names.append(f"aa_{metric}")
        names.extend([
            "gene_cosmic", "gene_mut_freq", "gene_length_norm",
        ])
        for chrom in CHROMOSOMES:
            names.append(f"chrom_{chrom}")
        names.append("pos_in_gene")
        self.feature_names = names

    # --- Encoding helpers ----------------------------------------------------

    @staticmethod
    def _encode_variant_type(
        df: pd.DataFrame, out: np.ndarray, offset: int
    ) -> int:
        n_cats = len(VARIANT_CATEGORIES) + 1  # +1 for Other
        col = df.get("variant_classification")
        if col is None:
            col = df.get("mutation_type")
        if col is None:
            return offset + n_cats

        for i, raw in enumerate(col):
            text = str(raw).strip() if pd.notna(raw) else ""
            canon = VARIANT_ALIASES.get(text.lower(), text)
            if canon in VARIANT_CATEGORIES:
                j = VARIANT_CATEGORIES.index(canon)
            else:
                j = len(VARIANT_CATEGORIES)  # Other
            out[i, offset + j] = 1.0
        return offset + n_cats

    @staticmethod
    def _encode_aa_properties(
        df: pd.DataFrame, out: np.ndarray, offset: int
    ) -> int:
        n_props = 5
        col = df.get("protein_change")
        if col is None:
            return offset + n_props

        for i, raw in enumerate(col):
            parsed = _parse_protein_change(raw)
            if parsed is None:
                continue
            ref, alt = parsed
            if ref == alt:
                continue
            out[i, offset] = _lookup_symmetric(GRANTHAM, ref, alt)
            out[i, offset + 1] = _lookup_symmetric(BLOSUM62, ref, alt)
            h_ref = AA_HYDROPHOBICITY.get(ref, 0.0)
            h_alt = AA_HYDROPHOBICITY.get(alt, 0.0)
            out[i, offset + 2] = h_alt - h_ref
            c_ref = AA_CHARGE.get(ref, 0.0)
            c_alt = AA_CHARGE.get(alt, 0.0)
            out[i, offset + 3] = c_alt - c_ref
            w_ref = AA_WEIGHT.get(ref, 0.0)
            w_alt = AA_WEIGHT.get(alt, 0.0)
            out[i, offset + 4] = w_alt - w_ref
        return offset + n_props

    def _encode_gene_features(
        self, df: pd.DataFrame, out: np.ndarray, offset: int
    ) -> int:
        n_gene = 3
        col = df.get("gene_symbol")
        if col is None:
            return offset + n_gene

        for i, raw in enumerate(col):
            gene = str(raw).strip().upper() if pd.notna(raw) else ""
            out[i, offset] = 1.0 if gene in COSMIC_CENSUS_GENES else 0.0
            freq = self._gene_freq.get(gene, 0.0)
            out[i, offset + 1] = freq / self._max_freq if self._max_freq else 0.0
            length = GENE_LENGTHS.get(gene, DEFAULT_GENE_LENGTH)
            out[i, offset + 2] = length / max(GENE_LENGTHS.values())
        return offset + n_gene

    @staticmethod
    def _encode_positional(
        df: pd.DataFrame, out: np.ndarray, offset: int
    ) -> int:
        n_chrom = len(CHROMOSOMES)
        chrom_col = df.get("chromosome")
        if chrom_col is not None:
            for i, raw in enumerate(chrom_col):
                text = str(raw).strip().upper() if pd.notna(raw) else ""
                if text.startswith("CHR"):
                    text = text[3:]
                if text in CHROMOSOMES:
                    j = CHROMOSOMES.index(text)
                    out[i, offset + j] = 1.0
        offset += n_chrom

        pos_col = df.get("start_position")
        gene_col = df.get("gene_symbol")
        if pos_col is not None and gene_col is not None:
            for i, (pos, gene) in enumerate(zip(pos_col, gene_col)):
                if pd.isna(pos) or pd.isna(gene):
                    continue
                gene_str = str(gene).strip().upper()
                length = GENE_LENGTHS.get(gene_str, DEFAULT_GENE_LENGTH)
                p = float(pos)
                norm = min(max(p / length, 0.0), 1.0) if length else 0.0
                out[i, offset] = norm
        return offset + 1
