"""Deterministic rule-based relationship extraction.

Extracts drug-target and other biomedical relationships from text using
linguistic patterns. Does NOT use LLMs - purely deterministic.
"""

import hashlib
import re
from dataclasses import dataclass
from typing import Any

from app.config import RELATIONSHIP_CONFIDENCE_THRESHOLD
from app.logging import logger
from app.nlp.biomedical_ner import ExtractedEntity, BiomedicalNER


@dataclass
class ExtractedRelationship:
    source_entity: ExtractedEntity
    target_entity: ExtractedEntity
    relationship_type: str
    confidence: float
    is_direct_interaction: bool
    negation: bool
    uncertainty: bool
    speculation: bool
    experimental_context: str | None
    source_passage: str
    source_passage_hash: str
    page_number: int | None = None
    section_name: str | None = None
    chunk_id: int | None = None
    extraction_method: str = "rule"


# Direct interaction predicate patterns - these indicate direct molecular interaction
DIRECT_INTERACTION_PREDICATES = {
    "inhibits", "activates", "binds_to", "upregulates", "downregulates",
    "regulates", "induces", "suppresses", "blocks", "antagonizes", "agonizes",
}

# Predicate keywords for matching
PREDICATE_KEYWORDS = [
    # (pattern, predicate_type, is_direct)
    (r"\binhibit\w*\b", "inhibits", True),
    (r"\bactivat\w*\b", "activates", True),
    (r"\bbind\w*\s+to\b", "binds_to", True),
    (r"\bupregulat\w*\b", "upregulates", True),
    (r"\bdownregulat\w*\b", "downregulates", True),
    (r"\bregulat\w*\b", "regulates", True),
    (r"\btreat\w*\b", "treats", False),
    (r"\bprevent\w*\b", "prevents", False),
    (r"\breduc\w*\b", "reduces", False),
    (r"\bincreas\w*\b", "increases", False),
    (r"\bmodulat\w*\b", "modulates", False),
    (r"\bassociat\w*\s+with\b", "associated_with", False),
    (r"\blink\w*\s+to\b", "linked_to", False),
    (r"\bcorrelat\w*\s+with\b", "correlated_with", False),
    (r"\binduc\w*\b", "induces", True),
    (r"\bsuppress\w*\b", "suppresses", True),
    (r"\bblock\w*\b", "blocks", True),
    (r"\bantagoniz\w*\b", "antagonizes", True),
    (r"\bagoniz\w*\b", "agonizes", True),
]

# Entity type compatibility: which predicate types work with which source/target combos
PREDICATE_ENTITY_COMPATIBILITY = {
    "inhibits": [("drug", "protein"), ("drug", "gene"), ("drug", "disease"),
                 ("protein", "protein"), ("gene", "protein")],
    "activates": [("drug", "protein"), ("drug", "gene"), ("protein", "protein")],
    "binds_to": [("drug", "protein"), ("drug", "gene"), ("protein", "protein")],
    "upregulates": [("drug", "protein"), ("drug", "gene"), ("protein", "protein")],
    "downregulates": [("drug", "protein"), ("drug", "gene"), ("protein", "protein")],
    "regulates": [("drug", "protein"), ("gene", "protein"), ("protein", "protein")],
    "treats": [("drug", "disease")],
    "associated_with": [("drug", "disease"), ("gene", "disease"), ("protein", "disease")],
}


class RelationshipExtractor:
    """Deterministic rule-based relationship extractor."""

    def __init__(self, ner: BiomedicalNER):
        self.ner = ner
        self._predicate_regexes = [
            (re.compile(p, re.IGNORECASE), pred, is_direct)
            for p, pred, is_direct in PREDICATE_KEYWORDS
        ]
        logger.info("Deterministic relationship extractor initialized")

    def extract_relationships(
        self,
        text: str,
        entities: list[ExtractedEntity],
        page_number: int | None = None,
        section_name: str | None = None,
        chunk_id: int | None = None,
    ) -> list[ExtractedRelationship]:
        """Extract relationships from text using entity co-occurrence and predicate patterns."""
        relationships: list[ExtractedRelationship] = []
        seen: set[tuple[str, str, str, str]] = set()  # (source_norm, target_norm, predicate, passage_hash)

        # Split text into sentences for context-aware extraction
        sentences = self._split_sentences(text)

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:
                continue

            # Detect context flags for this sentence
            context = self.ner.detect_context(sentence)

            # Find predicate in sentence
            predicate_match = self._find_predicate(sentence)
            if not predicate_match:
                continue

            predicate_type, is_direct = predicate_match

            # Find entities in sentence
            sentence_entities = [
                e for e in entities
                if e.entity_name_original.lower() in sentence.lower()
            ]

            if len(sentence_entities) < 2:
                continue

            # Try all source-target pairs
            for i, source_ent in enumerate(sentence_entities):
                for j, target_ent in enumerate(sentence_entities):
                    if i == j:
                        continue

                    # Check entity type compatibility
                    if not self._check_compatibility(
                        source_ent.entity_type, target_ent.entity_type, predicate_type
                    ):
                        continue

                    # Determine assertion polarity
                    source_passage = sentence
                    passage_hash = hashlib.sha256(source_passage.encode("utf-8")).hexdigest()

                    dedup_key = (
                        source_ent.normalized_name,
                        target_ent.normalized_name,
                        predicate_type,
                        passage_hash,
                    )
                    if dedup_key in seen:
                        continue
                    seen.add(dedup_key)

                    # Compute confidence
                    confidence = self._compute_confidence(
                        source_ent, target_ent, predicate_type, context, is_direct
                    )

                    if confidence < RELATIONSHIP_CONFIDENCE_THRESHOLD:
                        continue

                    relationships.append(
                        ExtractedRelationship(
                            source_entity=source_ent,
                            target_entity=target_ent,
                            relationship_type=predicate_type,
                            confidence=confidence,
                            is_direct_interaction=is_direct,
                            negation=context["negation"],
                            uncertainty=context["uncertainty"],
                            speculation=context["speculation"],
                            experimental_context=context["experimental_context"],
                            source_passage=source_passage,
                            source_passage_hash=passage_hash,
                            page_number=page_number,
                            section_name=section_name,
                            chunk_id=chunk_id,
                            extraction_method="rule",
                        )
                    )

        logger.info("Extracted %d relationships (threshold=%.2f)", len(relationships), RELATIONSHIP_CONFIDENCE_THRESHOLD)
        return relationships

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        # Simple sentence splitter
        sentences = re.split(r"[.!?]+(?:\s+|$)", text)
        return [s.strip() for s in sentences if s.strip()]

    def _find_predicate(
        self, text: str
    ) -> tuple[str, bool] | None:
        """Find a relationship predicate in text."""
        for pattern_regex, pred_type, is_direct in self._predicate_regexes:
            if pattern_regex.search(text):
                return (pred_type, is_direct)
        return None

    def _check_compatibility(
        self,
        source_type: str,
        target_type: str,
        predicate: str,
    ) -> bool:
        """Check if entity types are compatible with the predicate."""
        compatible_pairs = PREDICATE_ENTITY_COMPATIBILITY.get(predicate, [])
        # Check both directions
        for src, tgt in compatible_pairs:
            if (source_type == src and target_type == tgt) or \
               (source_type == tgt and target_type == src and predicate in ("associated_with", "linked_to", "correlated_with", "binds_to")):
                return True
        return False

    def _compute_confidence(
        self,
        source_ent: ExtractedEntity,
        target_ent: ExtractedEntity,
        predicate: str,
        context: dict,
        is_direct: bool,
    ) -> float:
        """Compute confidence score for a relationship."""
        base = 0.5

        # Boost for direct interactions
        if is_direct:
            base += 0.15

        # Boost for higher entity confidence
        base += (source_ent.confidence - 0.5) * 0.1
        base += (target_ent.confidence - 0.5) * 0.1

        # Reduce for negation
        if context["negation"]:
            base -= 0.2

        # Reduce for uncertainty
        if context["uncertainty"]:
            base -= 0.1

        # Reduce for speculation
        if context["speculation"]:
            base -= 0.1

        # Boost for experimental context
        if context["experimental_context"]:
            base += 0.05

        return max(0.0, min(1.0, base))
