"""El catálogo YAML y el dict embebido no pueden divergir.

model_catalog.yaml es la fuente en runtime; BUILTIN_CATALOG es el fallback.
Si alguien agrega un modelo en un solo lado, estos tests lo cazan.
"""

import dataclasses

import pytest

from engines.base import (
    BUILTIN_CATALOG,
    MODEL_CATALOG,
    catalog_candidates,
    load_catalog,
)


def _yaml_source():
    for candidate in catalog_candidates():
        if candidate.is_file():
            return candidate
    return None


class TestCatalogYaml:

    def test_repo_ships_the_catalog_yaml(self):
        assert _yaml_source() is not None, "falta model_catalog.yaml en el repo"

    def test_yaml_matches_builtin_catalog(self):
        pytest.importorskip("yaml")
        source = _yaml_source()
        assert source is not None

        yaml_catalog = load_catalog(source)

        assert yaml_catalog.keys() == BUILTIN_CATALOG.keys()
        for key, expected in BUILTIN_CATALOG.items():
            assert dataclasses.asdict(yaml_catalog[key]) == dataclasses.asdict(expected), key

    def test_runtime_catalog_is_loaded_at_import(self):
        pytest.importorskip("yaml")
        assert MODEL_CATALOG.keys() == BUILTIN_CATALOG.keys()

    def test_missing_file_falls_back_to_builtin(self, tmp_path):
        catalog = load_catalog(tmp_path / "no-existe.yaml")
        assert catalog.keys() == BUILTIN_CATALOG.keys()

    def test_invalid_yaml_falls_back_to_builtin(self, tmp_path):
        pytest.importorskip("yaml")
        bad = tmp_path / "model_catalog.yaml"
        bad.write_text("models: [esto, no, es, un, mapa]\n", encoding="utf-8")

        catalog = load_catalog(bad)

        assert catalog.keys() == BUILTIN_CATALOG.keys()

    def test_invalid_entry_is_skipped_without_dropping_the_rest(self, tmp_path):
        pytest.importorskip("yaml")
        catalog_file = tmp_path / "model_catalog.yaml"
        catalog_file.write_text(
            "models:\n"
            "  Ok-Mock:\n"
            "    label: \"Ok mock (test)\"\n"
            "    engine: mock\n"
            "    mode: image\n"
            "    pipeline: sd15\n"
            "    family: sd\n"
            "  Sin-Engine:\n"
            "    mode: image\n"
            "    pipeline: sd15\n"
            "    family: sd\n",
            encoding="utf-8",
        )

        catalog = load_catalog(catalog_file)

        assert set(catalog) == {"Ok-Mock"}
        assert catalog["Ok-Mock"].label == "Ok mock (test)"
