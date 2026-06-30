from __future__ import annotations

import re
from dataclasses import dataclass

from whichvlm.models.types import ModelInfo


@dataclass(frozen=True)
class IntegrationProfile:
    integration_id: str
    capabilities: tuple[str, ...]
    pipeline_tags: tuple[str, ...]
    tag_patterns: tuple[str, ...]
    component_roles: tuple[str, ...]


INTEGRATION_PROFILES: tuple[IntegrationProfile, ...] = (
    IntegrationProfile(
        integration_id="vision-language",
        capabilities=("vision",),
        pipeline_tags=(
            "image-text-to-text",
            "visual-question-answering",
            "image-to-text",
        ),
        tag_patterns=(
            r"(^|[-_/\s])(vl|vision|multimodal|llava|image)([-_/\s]|$)",
            r"image-text-to-text|visual-question-answering|image-to-text",
            r"qwen.*vl|internvl|pixtral|deepseek-vl",
        ),
        component_roles=("language", "vision_encoder", "projector", "processor"),
    ),
    IntegrationProfile(
        integration_id="document-ocr",
        capabilities=("vision", "ocr"),
        pipeline_tags=(),
        tag_patterns=(
            r"(^|[-_/\s])(ocr|docvqa|document)([-_/\s]|$)",
            r"text[-_ ]?recognition",
        ),
        component_roles=("language", "vision_encoder", "projector", "processor"),
    ),
)


def _append_unique(values: list[str], candidates: tuple[str, ...]) -> None:
    for candidate in candidates:
        if candidate not in values:
            values.append(candidate)


def _matches_profile(
    profile: IntegrationProfile,
    model_id: str,
    pipeline_tag: object,
    tags: list[str],
    architecture: str = "",
    component_roles: tuple[str, ...] = (),
) -> bool:
    if pipeline_tag in profile.pipeline_tags:
        return True
    if profile.integration_id == "vision-language" and any(
        role in profile.component_roles and role != "language"
        for role in component_roles
    ):
        return True
    haystack = " ".join(
        [model_id, str(pipeline_tag or ""), *tags, architecture]
    ).lower()
    return any(re.search(pattern, haystack) for pattern in profile.tag_patterns)


def capability_ids_for_data(
    model_id: str,
    pipeline_tag: object,
    tags: list[str],
    architecture: str = "",
    component_roles: tuple[str, ...] = (),
) -> list[str]:
    capabilities: list[str] = []
    for profile in INTEGRATION_PROFILES:
        if _matches_profile(
            profile,
            model_id,
            pipeline_tag,
            tags,
            architecture,
            component_roles,
        ):
            _append_unique(capabilities, profile.capabilities)
    return capabilities


def capability_ids_for_model(model: ModelInfo) -> list[str]:
    component_roles = tuple(component.role for component in model.components)
    capabilities = capability_ids_for_data(
        model.id,
        model.hf_pipeline_tag,
        model.tags,
        model.architecture,
        component_roles,
    )
    _append_unique(capabilities, tuple(model.capabilities))
    return capabilities


def has_capability(model: ModelInfo, capability: str) -> bool:
    return capability in capability_ids_for_model(model)


def component_roles_for_capabilities(capabilities: list[str]) -> list[str]:
    roles: list[str] = []
    for profile in INTEGRATION_PROFILES:
        if any(capability in capabilities for capability in profile.capabilities):
            _append_unique(roles, profile.component_roles)
    return roles


def pipeline_tags_for_capability(capability: str) -> tuple[str, ...]:
    tags: list[str] = []
    for profile in INTEGRATION_PROFILES:
        if capability in profile.capabilities:
            _append_unique(tags, profile.pipeline_tags)
    return tuple(tags)
