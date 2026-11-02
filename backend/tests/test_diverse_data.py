"""Modern-data partitions preserve prompt groups and source/model balance."""

from __future__ import annotations

import pytest

from veritas.data.diverse import prompt_group, select_rows


def row(identifier, label, model, prompt=None):
    return {
        "row_id": identifier,
        "label": label,
        "model": model,
        "prompt": prompt if prompt is not None else identifier,
        "image_sha256": identifier.encode().hex().ljust(64, "0")[:64],
    }


def test_prompt_variants_share_a_group_across_generators():
    a = row("a", "real", "pexels", "  Café  by the sea ")
    b = row("b", "fake", "flux.2-dev", "CAFÉ by the sea")
    assert prompt_group(a) == prompt_group(b)
    assert prompt_group(a) != prompt_group(row("c", "fake", "sd-3.5", "a city at night"))


def test_selection_keeps_both_real_sources_and_multiple_fake_models():
    candidates = [row(str(i), "real", source) for i, source in enumerate(["laion"] * 3 + ["pexels"] * 3)]
    candidates += [
        row(str(i + 10), "fake", model) for i, model in enumerate(["flux.2-dev"] * 3 + ["sd-3.5"] * 3)
    ]
    selected = select_rows(
        candidates, real_quotas={"laion": 2, "pexels": 2}, fake_count=4, fake_model_cap=2, seed=7
    )
    assert len(selected) == 8
    assert sum(r["label"] == "real" and r["model"] == "laion" for r in selected) == 2
    assert sum(r["label"] == "fake" and r["model"] == "flux.2-dev" for r in selected) == 2
    assert (
        select_rows(
            list(reversed(candidates)),
            real_quotas={"laion": 2, "pexels": 2},
            fake_count=4,
            fake_model_cap=2,
            seed=7,
        )
        == selected
    )


def test_held_out_prompt_cannot_enter_training_under_another_model_or_filename():
    reserved = row("reserved", "real", "docci", "same source scene")
    candidate = row("different", "fake", "flux.2-dev", " SAME source scene ")
    pool = [candidate, row("ok-real", "real", "pexels"), row("ok-fake", "fake", "sd-3.5")]
    selected = select_rows(
        pool,
        real_quotas={"pexels": 1},
        fake_count=1,
        fake_model_cap=1,
        excluded_groups={prompt_group(reserved)},
        seed=0,
    )
    assert {r["row_id"] for r in selected} == {"ok-real", "ok-fake"}


def test_incomplete_source_or_generator_coverage_is_rejected():
    pool = [row("a", "real", "laion"), row("b", "fake", "flux.2-dev")]
    with pytest.raises(ValueError, match="quota"):
        select_rows(pool, real_quotas={"laion": 1, "pexels": 1}, fake_count=2, fake_model_cap=1)


def test_same_original_image_is_not_counted_twice():
    a = row("a", "real", "pexels")
    b = row("b", "real", "pexels")
    b["image_sha256"] = a["image_sha256"]
    with pytest.raises(ValueError, match="quota"):
        select_rows(
            [a, b, row("c", "fake", "flux.2-dev")], real_quotas={"pexels": 2}, fake_count=1, fake_model_cap=1
        )
