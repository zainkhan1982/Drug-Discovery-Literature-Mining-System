"""Biomedical Named Entity Recognition.

Uses a local rule-based/dictionary approach that works without external model
downloads. Structured so SciSpaCy models can be swapped in later.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from app.config import ENTITY_CONFIDENCE_THRESHOLD
from app.logging import logger


@dataclass
class ExtractedEntity:
    entity_name_original: str
    normalized_name: str
    entity_type: str
    confidence: float
    page_number: Optional[int] = None
    section_name: Optional[str] = None
    chunk_id: Optional[int] = None
    extraction_method: str = "rule"
    external_identifier: Optional[str] = None
    external_identifier_source: Optional[str] = None


# Curated biomedical dictionaries for rule-based NER
DRUG_KEYWORDS = {
    "aspirin", "ibuprofen", "acetaminophen", "paracetamol", "warfarin", "heparin",
    "atorvastatin", "simvastatin", "lisinopril", "metformin", "insulin",
    "doxorubicin", "cisplatin", "paclitaxel", "imatinib", "trastuzumab",
    "adalimumab", "etanercept", "infliximab", "rituximab", "bevacizumab",
    "omeprazole", "loratadine", "cetirizine", "amlodipine", "metoprolol",
    "albuterol", "fluticasone", "montelukast", "warfarin", "clopidogrel",
    "apixaban", "rivaroxaban", "dabigatran", "levothyroxine", "prednisone",
    "hydrocortisone", "dexamethasone", "morphine", "fentanyl", "oxycodone",
    "gabapentin", "pregabalin", "sertraline", "fluoxetine", "citalopram",
    "venlafaxine", "bupropion", "lorazepam", "diazepam", "alprazolam",
    "donepezil", "memantine", "rivastigmine", "levodopa", "carbidopa",
}

GENE_KEYWORDS = {
    "brca1", "brca2", "tp53", "egfr", "her2", "kras", "nras", "alk", "ros1",
    "pten", "apc", "ret", "met", "cdk4", "cdk6", "myc", "mycn", "idh1",
    "idh2", "flt3", "npm1", "cebpa", "wt1", "smarcb1", "vhl", "nf1", "nf2",
    "p53", "ras", "src", "abl", "myc", "erbB2", "vegf", "vegfr",
    "mmp", "mmp2", "mmp9", "ctnnb1", "ctnna1", "apc", "axin",
}

PROTEIN_KEYWORDS = {
    "albumin", "globulin", "hemoglobin", "insulin", "glucagon",
    "interleukin", "interferon", "tumor necrosis factor", "tnf",
    "transforming growth factor", "tgf", "epidermal growth factor", "egf",
    "fibroblast growth factor", "fgf", "vascular endothelial growth factor", "vegf",
    "toll-like receptor", "tlr", "major histocompatibility complex", "mhc",
    "immunoglobulin", "igg", "iga", "igm", "ige", "igd",
    "cd4", "cd8", "cd19", "cd20", "cd25", "cd34",
    "caspase", "cyclin", "cdk", "p21", "p27", "rb",
    "akt", "mapk", "erk", "jak", "stat", "pi3k", "mTOR",
    "bcl-2", "bax", "bak", "bim", "puffa",
}

DISEASE_KEYWORDS = {
    "alzheimer", "parkinson", "huntington", "als", "multiple sclerosis",
    "diabetes", "cancer", "carcinoma", "sarcoma", "lymphoma", "leukemia",
    "melanoma", "astrocytoma", "glioblastoma", "breast cancer",
    "prostate cancer", "lung cancer", "colorectal cancer",
    "hypertension", "stroke", "myocardial infarction", "heart failure",
    "atherosclerosis", "copd", "asthma", "arthritis", "rheumatoid arthritis",
    "osteoarthritis", "lupus", "psoriasis", "eczema",
    "depression", "schizophrenia", "bipolar", "anxiety", "autism",
    "epilepsy", "migraine", "neuropathy", "nephritis", "hepatitis",
    "cirrhosis", "fibrosis", "osteoporosis", "obesity", "hyperlipidemia",
}

PATHWAY_KEYWORDS = {
    "mapk pathway", "pi3k-akt pathway", "vegf pathway", "egfr pathway",
    "wnt pathway", "notch pathway", "hedgehog pathway", "jak-stat pathway",
    "nf-kb pathway", "p53 pathway", "cell cycle pathway",
    "apoptosis pathway", "dna repair pathway", "metabolic pathway",
    "krebs cycle", "glycolysis", "oxidative phosphorylation",
    "complement pathway", "coagulation cascade",
}

# Negation/uncertainty/speculation patterns
NEGATION_PATTERNS = [
    r"\bno\s+\w+", r"\bnot\s+\w+", r"\bnever\s+\w+", r"\bnone\s+\w+",
    r"\bdoes\s+not\s+\w+", r"\bdoesn't\s+\w+", r"\bdo\s+not\s+\w+",
    r"\bdid\s+not\s+\w+", r"\bdidn't\s+\w+", r"\bwithout\s+\w+",
    r"\black\s+of\s+\w+", r"\babsence\s+of\s+\w+", r"\bneither\s+\w+",
    r"\bnor\s+\w+", r"\brule\s+out\b", r"\brules?\s+out\b",
    r"\bdenies?\s+\w+", r"\bnegative\s+for\b", r"\bnot\s+associated\b",
    r"\bnot\s+correlated\b", r"\bnot\s+linked\b", r"\bnot\s+found\b",
]

UNCERTAINTY_PATTERNS = [
    r"\bmay\b", r"\bmight\b", r"\bcould\b", r"\bpossible\b", r"\bpossibly\b",
    r"\bprobable\b", r"\bprobably\b", r"\blikely\b", r"\bunlikely\b",
    r"\bsuggest\b", r"\bsuggests\b", r"\bsuggestion\b",
    r"\bappears?\s+to\b", r"\bseems?\s+to\b", r"\bpotential\b",
    r"\bpotentially\b", r"\bpresum\w+\b", r"\btentativ\w+\b",
    r"\bunclear\b", r"\buncertain\b", r"\bequivocal\b",
    r"\binconclusive\b", r"\bnot\s+clear\b", r"\bnot\s+established\b",
]

SPECULATION_PATTERNS = [
    r"\bhypothes\w+\b", r"\bhypothesis\b", r"\bpropos\w+\b",
    r"\bspeculat\w+\b", r"\bconjectur\w+\b", r"\btheoriz\w+\b",
    r"\bmay\s+suggest\b", r"\bmight\s+indicate\b",
    r"\bcould\s+represent\b", r"\bpotential\s+role\b",
    r"\bpossible\s+role\b", r"\bmay\s+play\s+a\s+role\b",
    r"\bmay\s+contribute\b", r"\bmay\s+involve\b",
]

# Experimental context patterns
EXPERIMENTAL_CONTEXT_PATTERNS = [
    (r"\bin\s+vitro\b", "in_vitro"),
    (r"\bin\s+vivo\b", "in_vivo"),
    (r"\banimal\s+model\b", "in_vivo"),
    (r"\bmouse\b", "in_vivo"),
    (r"\brat\b", "in_vivo"),
    (r"\bcell\s+line\b", "cell_line"),
    (r"\bcultured\b", "cell_line"),
    (r"\bprimary\s+cell\b", "cell_line"),
    (r"\bclinical\s+trial\b", "clinical"),
    (r"\bpatient\b", "clinical"),
    (r"\bpatients\b", "clinical"),
    (r"\bhuman\b", "clinical"),
    (r"\bretrospective\b", "clinical"),
    (r"\bprospective\b", "clinical"),
    (r"\brandomized\b", "clinical"),
]

# Relationship predicate patterns: subject - PREDICATE - object
PREDICATE_PATTERNS = [
    # Direct interactions
    (r"\binhibit\w*\s+", "inhibits"),
    (r"\bactivat\w*\s+", "activates"),
    (r"\bbind\w*\s+to\s+", "binds_to"),
    (r"\bupregulat\w*\s+", "upregulates"),
    (r"\bdownregulat\w*\s+", "downregulates"),
    (r"\bregulat\w*\s+", "regulates"),
    (r"\btreat\w*\s+", "treats"),
    (r"\bprevent\w*\s+", "prevents"),
    (r"\breduc\w*\s+", "reduces"),
    (r"\bincreas\w*\s+", "increases"),
    (r"\bmodulat\w*\s+", "modulates"),
    (r"\bassociat\w*\s+with\s+", "associated_with"),
    (r"\blink\w*\s+to\s+", "linked_to"),
    (r"\bcorrelat\w*\s+with\s+", "correlated_with"),
    (r"\binduc\w*\s+", "induces"),
    (r"\bsuppress\w*\s+", "suppresses"),
    (r"\bblock\w*\s+", "blocks"),
    (r"\bantagoniz\w*\s+", "antagonizes"),
    (r"\bagoniz\w*\s+", "agonizes"),
]

ENTITY_TYPE_KEYWORDS = {
    "drug": DRUG_KEYWORDS,
    "gene": GENE_KEYWORDS,
    "protein": PROTEIN_KEYWORDS,
    "disease": DISEASE_KEYWORDS,
    "pathway": PATHWAY_KEYWORDS,
}


class BiomedicalNER:
    """Biomedical NER using rule-based dictionary matching with optional SciSpaCy backend."""

    def __init__(self, use_scispacy: bool = False):
        self.use_scispacy = use_scispacy and self._check_scispacy()
        self._scispacy_models: dict[str, Any] = {}
        if self.use_scispacy:
            self._load_scispacy_models()
        logger.info("Biomedical NER initialized (scispacy=%s)", self.use_scispacy)

    def _check_scispacy(self) -> bool:
        try:
            import spacy  # noqa: F401
            return True
        except ImportError:
            return False

    def _load_scispacy_models(self) -> None:
        try:
            import spacy
            from app.config import (
                SCISPACY_CHEMICAL_DISEASE_MODEL,
                SCISPACY_GENE_PROTEIN_MODEL,
            )
            self._scispacy_models["chemical_disease"] = spacy.load(SCISPACY_CHEMICAL_DISEASE_MODEL)
            self._scispacy_models["gene_protein"] = spacy.load(SCISPACY_GENE_PROTEIN_MODEL)
            logger.info("SciSpaCy models loaded successfully")
        except Exception as exc:  # noqa: BLE001
            logger.warning("SciSpaCy models not available, using rule-based: %s", exc)
            self.use_scispacy = False

    def extract_entities(
        self,
        text: str,
        page_number: int | None = None,
        section_name: str | None = None,
        chunk_id: int | None = None,
    ) -> list[ExtractedEntity]:
        """Extract biomedical entities from text."""
        entities: list[ExtractedEntity] = []
        seen: set[tuple[str, str]] = set()  # (normalized, type) dedup

        # Rule-based dictionary matching
        for entity_type, keywords in ENTITY_TYPE_KEYWORDS.items():
            for keyword in keywords:
                # Match whole words/phrases, case-insensitive
                pattern = r"\b" + re.escape(keyword) + r"\b"
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    original = match.group(0)
                    normalized = original.lower()
                    key = (normalized, entity_type)
                    if key in seen:
                        continue
                    seen.add(key)

                    entities.append(
                        ExtractedEntity(
                            entity_name_original=original,
                            normalized_name=normalized,
                            entity_type=entity_type,
                            confidence=0.7,  # Rule-based confidence
                            page_number=page_number,
                            section_name=section_name,
                            chunk_id=chunk_id,
                            extraction_method="rule",
                        )
                    )

        # SciSpaCy enhancement (if available)
        if self.use_scispacy:
            try:
                scispacy_entities = self._extract_with_scispacy(text, page_number, section_name, chunk_id)
                for se in scispacy_entities:
                    key = (se.normalized_name, se.entity_type)
                    if key not in seen:
                        seen.add(key)
                        entities.append(se)
            except Exception as exc:  # noqa: BLE001
                logger.warning("SciSpaCy extraction failed: %s", exc)

        # Filter by confidence threshold
        entities = [e for e in entities if e.confidence >= ENTITY_CONFIDENCE_THRESHOLD]

        logger.info("Extracted %d entities (threshold=%.2f)", len(entities), ENTITY_CONFIDENCE_THRESHOLD)
        return entities

    def _extract_with_scispacy(
        self,
        text: str,
        page_number: int | None,
        section_name: str | None,
        chunk_id: int | None,
    ) -> list[ExtractedEntity]:
        """Extract entities using SciSpaCy models."""
        import spacy  # type: ignore

        entities: list[ExtractedEntity] = []
        seen: set[tuple[str, str]] = set()

        for model_name, nlp in self._scispacy_models.items():
            doc = nlp(text)
            for ent in doc.ents:
                normalized = ent.text.lower()
                entity_type = self._map_scispacy_label(ent.label_, model_name)
                key = (normalized, entity_type)
                if key in seen:
                    continue
                seen.add(key)
                entities.append(
                    ExtractedEntity(
                        entity_name_original=ent.text,
                        normalized_name=normalized,
                        entity_type=entity_type,
                        confidence=0.85,  # SciSpaCy confidence
                        page_number=page_number,
                        section_name=section_name,
                        chunk_id=chunk_id,
                        extraction_method="scispacy",
                    )
                )
        return entities

    def _map_scispacy_label(self, label: str, model_name: str) -> str:
        """Map SciSpaCy entity labels to our schema."""
        label = label.upper()
        mapping = {
            "CHEMICAL": "drug",
            "DISEASE": "disease",
            "GENE_OR_GENE_PRODUCT": "gene",
            "GENE": "gene",
            "PROTEIN": "protein",
        }
        return mapping.get(label, "concept")

    def normalize_entity(self, name: str) -> str:
        """Normalize entity name: lowercase, strip punctuation, collapse whitespace."""
        if not name:
            return ""
        n = name.lower()
        n = re.sub(r"[^\w\s]", "", n)
        n = re.sub(r"\s+", " ", n).strip()
        return n

    def detect_context(
        self,
        text: str,
    ) -> dict[str, bool | str | None]:
        """Detect negation, uncertainty, speculation, and experimental context."""
        text_lower = text.lower()

        negation = any(re.search(p, text_lower) for p in NEGATION_PATTERNS)
        uncertainty = any(re.search(p, text_lower) for p in UNCERTAINTY_PATTERNS)
        speculation = any(re.search(p, text_lower) for p in SPECULATION_PATTERNS)

        experimental_context = None
        for pattern, context in EXPERIMENTAL_CONTEXT_PATTERNS:
            if re.search(pattern, text_lower):
                experimental_context = context
                break

        return {
            "negation": bool(negation),
            "uncertainty": bool(uncertainty),
            "speculation": bool(speculation),
            "experimental_context": experimental_context,
        }
