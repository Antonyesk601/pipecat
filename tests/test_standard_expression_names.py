#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the names TTS services perform the standard deliveries and sounds by."""

import pytest

from pipecat.adapters.tts.services.cartesia_adapter import CartesiaExpressionAdapter
from pipecat.adapters.tts.services.elevenlabs_adapter import ElevenLabsAudioTagExpressionAdapter
from pipecat.adapters.tts.services.fish_adapter import FishAudioS2ExpressionAdapter
from pipecat.adapters.tts.services.minimax_adapter import MiniMaxExpressionAdapter
from pipecat.expressive.expression import (
    STANDARD_DELIVERY,
    STANDARD_SOUNDS,
    ExpressionCapabilities,
    ExpressionTerm,
    standard_name,
)

ADAPTERS = [
    CartesiaExpressionAdapter,
    ElevenLabsAudioTagExpressionAdapter,
    FishAudioS2ExpressionAdapter,
    MiniMaxExpressionAdapter,
]


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_every_standard_name_an_adapter_lists_is_one_it_performs(adapter):
    capabilities = adapter().capabilities()

    assert set(capabilities.standard_names) <= {*STANDARD_DELIVERY, *STANDARD_SOUNDS}
    for label, names in capabilities.standard_names.items():
        for name in names:
            assert standard_name(capabilities, label, offered=[name]) == name


def test_the_best_name_the_policy_offers_is_chosen():
    capabilities = ExpressionCapabilities(
        delivery=[ExpressionTerm("calm"), ExpressionTerm("reassuring")],
        standard_names={"calm": ["calm", "reassuring"]},
    )

    assert standard_name(capabilities, "calm") == "calm"
    assert standard_name(capabilities, "calm", offered=["reassuring", "happy"]) == "reassuring"
    assert standard_name(capabilities, "calm", offered=["happy"]) is None


def test_a_label_left_out_is_performed_by_its_own_name_when_the_service_declares_it():
    capabilities = ExpressionCapabilities(
        sounds=[ExpressionTerm("laughter")],
        delivery=[ExpressionTerm("calm")],
        standard_names={"laugh": ["laughter"]},
    )

    assert standard_name(capabilities, "calm") == "calm"
    assert standard_name(capabilities, "laugh") == "laughter"
    assert standard_name(capabilities, "sad") is None
    assert standard_name(capabilities, "sigh") is None


def test_a_service_that_takes_free_text_also_takes_the_label_last():
    capabilities = ExpressionCapabilities(
        free_text_sounds=True, free_text_delivery=True, standard_names={"warm": ["warmly"]}
    )

    assert standard_name(capabilities, "grateful") == "grateful"
    assert standard_name(capabilities, "warm") == "warmly"
    assert standard_name(capabilities, "warm", offered=["warm"]) == "warm"


def test_a_name_the_service_doesnt_perform_is_passed_over():
    capabilities = ExpressionCapabilities(
        delivery=[ExpressionTerm("happy")], standard_names={"warm": ["content", "happy"]}
    )

    assert standard_name(capabilities, "warm") == "happy"
