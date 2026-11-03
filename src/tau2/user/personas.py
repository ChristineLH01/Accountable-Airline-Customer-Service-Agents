"""English caller personas for the text user simulator.

Every simulated caller speaks with one of two reviewed English personas taken
from the tau-bench 2.0 English language pack. A persona adds a gender line and
a few "pragmatics" clauses (register, how the caller opens, pushes back and
closes) to the user simulator's system prompt. The task scenario still owns the
caller's identity and goals; the persona only shapes how the caller speaks.

Persona data and the per-task caller genders live in
``data/tau2-<domain>/user_personas.json``. Each task's persona is
chosen deterministically: among the personas matching the task's caller
gender, the one selected by a SHA-256 of ``"{seed}:{task_id}"`` (the run
seed, not the per-trial seed). The English pack has exactly one persona per
gender, so every airline task always gets the same persona.
"""

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

from tau2.data_model.persona import PersonaConfig
from tau2.utils.utils import domain_data_dir

Gender = Literal["male", "female"]


class UserPersona(PersonaConfig):
    """A named caller persona: runtime persona settings plus speaking style."""

    persona_id: str = Field(description="Stable identifier of the persona.")
    gender: Gender = Field(description="Gender of the caller.")
    gender_guidance: str = Field(description="Gender line shown to the simulator.")
    pragmatics_clauses: list[str] = Field(
        default_factory=list,
        description="Reviewed speaking-style clauses, rendered in order.",
    )

    def to_guidelines_text(self) -> Optional[str]:
        """Persona behavior inserted into the simulator scaffold."""
        sections = [
            super().to_guidelines_text() or "",
            self.gender_guidance,
            *self.pragmatics_clauses,
        ]
        return "\n\n".join(s for s in sections if s) or None


class PersonaCatalog(BaseModel):
    """Personas available to a domain and the caller gender of each task."""

    source: str = Field(default="", description="Provenance of the persona data.")
    personas: dict[str, UserPersona] = Field(description="Personas by ID.")
    caller_genders: dict[str, Optional[Gender]] = Field(
        default_factory=dict,
        description="Caller gender of each task ID; tasks not listed match any persona.",
    )

    def assign(self, task_id: str, seed: Optional[int]) -> UserPersona:
        """Deterministic, gender-matched persona for a task."""
        gender = self.caller_genders.get(task_id)
        eligible = sorted(
            pid
            for pid, persona in self.personas.items()
            if gender is None or persona.gender == gender
        )
        if not eligible:
            raise ValueError(f"No persona matches caller gender {gender!r}")
        digest = hashlib.sha256(f"{seed or 0}:{task_id}".encode()).digest()
        index = int.from_bytes(digest[:8], "big") % len(eligible)
        return self.personas[eligible[index]]


def persona_catalog_path(domain: str) -> Path:
    """Location of a domain's persona catalog."""
    return domain_data_dir(domain) / "user_personas.json"


@lru_cache(maxsize=None)
def load_persona_catalog(domain: str) -> Optional[PersonaCatalog]:
    """Load a domain's persona catalog; None if the domain ships none."""
    path = persona_catalog_path(domain)
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    for persona_id, persona in data.get("personas", {}).items():
        persona.setdefault("persona_id", persona_id)
    return PersonaCatalog.model_validate(data)


def assign_user_persona(
    domain: str, task_id: str, seed: Optional[int]
) -> Optional[UserPersona]:
    """The persona of a task's caller, or None for domains without personas."""
    catalog = load_persona_catalog(domain)
    if catalog is None:
        return None
    return catalog.assign(task_id, seed)


def get_user_persona(domain: str, persona_id: str) -> UserPersona:
    """Look up a persona by ID (e.g. one recorded in saved results)."""
    catalog = load_persona_catalog(domain)
    if catalog is None or persona_id not in catalog.personas:
        raise ValueError(f"Domain {domain!r} has no user persona {persona_id!r}")
    return catalog.personas[persona_id]
