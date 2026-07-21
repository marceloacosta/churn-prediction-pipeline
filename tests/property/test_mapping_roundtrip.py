"""
Property 1: Mapping Config Round-Trip Consistency
==================================================

For ANY valid mapping configuration (with arbitrary client_id, column_mappings,
value_mappings, and type_coercions), serializing it to YAML and then parsing it
back SHALL produce a structurally equivalent MappingConfig object.

This is a "round-trip" property — like translating English to Spanish and back.
If the translation is correct, you get back what you started with. If
serialization or parsing has bugs, the round-trip breaks.

Validates: Requirements 2.5, 2.6

# Feature: churn-prediction-pipeline, Property 1: Mapping Config Round-Trip Consistency
"""

import tempfile
import os

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from churn_pipeline.mapping_config import (
    MappingConfig,
    load_mapping_config,
    serialize_mapping_config,
)


# ---------------------------------------------------------------------------
# Hypothesis strategies for generating random but valid MappingConfig objects
# ---------------------------------------------------------------------------

# Valid identifiers: non-empty strings without special YAML characters
_identifier = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="_-"),
    min_size=1,
    max_size=30,
).filter(lambda s: s[0].isalpha())

# Column names: simple identifiers that won't break YAML
_column_name = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="_"),
    min_size=1,
    max_size=20,
).filter(lambda s: s[0].isalpha())

# Values that can appear in value_mappings: strings, ints, floats
_mapping_value = st.one_of(
    st.text(min_size=1, max_size=15, alphabet=st.characters(whitelist_categories=("L", "N"))),
    st.integers(min_value=-100, max_value=100),
    st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
)

# Type coercion targets
_type_target = st.sampled_from(["float", "int", "string", "category"])

# Full MappingConfig strategy
_mapping_config = st.builds(
    MappingConfig,
    client_id=_identifier,
    source_description=st.text(
        min_size=0,
        max_size=50,
        alphabet=st.characters(whitelist_categories=("L", "N", "Z"), whitelist_characters=" _-,."),
    ),
    column_mappings=st.dictionaries(
        keys=_column_name,
        values=_column_name,
        min_size=0,
        max_size=10,
    ),
    value_mappings=st.dictionaries(
        keys=_column_name,
        values=st.dictionaries(
            keys=st.text(min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=("L", "N"))),
            values=_mapping_value,
            min_size=1,
            max_size=5,
        ),
        min_size=0,
        max_size=5,
    ),
    type_coercions=st.dictionaries(
        keys=_column_name,
        values=_type_target,
        min_size=0,
        max_size=5,
    ),
)


@pytest.mark.property
@given(config=_mapping_config)
@settings(max_examples=200)
def test_mapping_config_roundtrip(config: MappingConfig) -> None:
    """
    Serialize a random MappingConfig to YAML, write to disk, parse back,
    and verify structural equivalence.
    """
    # Serialize to YAML string
    yaml_str = serialize_mapping_config(config)

    # Write to a temp file and parse back
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_str)
        tmp_path = f.name

    try:
        loaded = load_mapping_config(tmp_path)
    finally:
        os.unlink(tmp_path)

    # Verify structural equivalence
    assert loaded.client_id == config.client_id

    # source_description: empty strings get dropped during serialization
    if config.source_description:
        assert loaded.source_description == config.source_description
    else:
        assert loaded.source_description == ""

    # column_mappings: empty dicts get dropped during serialization
    if config.column_mappings:
        assert loaded.column_mappings == config.column_mappings
    else:
        assert loaded.column_mappings == {}

    # value_mappings: check structural equivalence
    # YAML may convert numeric string keys, so we compare semantically
    if config.value_mappings:
        assert set(loaded.value_mappings.keys()) == set(config.value_mappings.keys())
        for field_name in config.value_mappings:
            orig_map = config.value_mappings[field_name]
            loaded_map = loaded.value_mappings[field_name]
            # Compare each key-value pair, accounting for YAML type round-trips
            for key in orig_map:
                # YAML may represent string keys differently
                loaded_key = str(key) if str(key) in loaded_map else key
                assert loaded_key in loaded_map, (
                    f"Key {key!r} missing from loaded value_mappings[{field_name}]"
                )
                orig_val = orig_map[key]
                loaded_val = loaded_map[loaded_key]
                # Floats may lose precision in YAML, compare approximately
                if isinstance(orig_val, float):
                    assert abs(loaded_val - orig_val) < 1e-6, (
                        f"Float mismatch: {orig_val} vs {loaded_val}"
                    )
                else:
                    assert loaded_val == orig_val
    else:
        assert loaded.value_mappings == {}

    # type_coercions: empty dicts get dropped during serialization
    if config.type_coercions:
        assert loaded.type_coercions == config.type_coercions
    else:
        assert loaded.type_coercions == {}
