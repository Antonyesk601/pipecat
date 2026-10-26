#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""The processor that reads how each user turn was meant with Pipecat Tone."""

import asyncio
from typing import Literal

from pipecat.expressive.perception import PerceptionCapabilities, PerceptionSignal, PerceptionTerm
from pipecat.frames.frames import (
    Frame,
    LLMContextAssistantTurnFrame,
    PerceptionMetadataFrame,
    StartFrame,
    TranscriptionFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.tone.reader.model import PipecatToneReaderModel

_EMOTION = (
    "The feeling the words show. Read from the words alone, so neutral means they show no "
    "strong feeling, not that the user feels none: go by what they say."
)
_SARCASM = (
    "The chance the turn is sarcastic, from 0 to 1. Read the emotion together with it: "
    "a sarcastic turn can read happy."
)


class PipecatToneReader(FrameProcessor):
    """Reads how each user turn was meant with Pipecat Tone.

    Goes between the STT service and the user aggregator. Each final
    transcription is read, with the bot's reply before it, and the reading is
    set as its ``perception``: the turn's ``emotion``, its two likeliest labels
    with their confidences, and its ``sarcasm``, the chance it's sarcastic.
    The read runs off the event loop, and the transcription waits for it, a
    few milliseconds.

    Part of expressive conversation (alpha): turn it on with
    ``PipelineParams(enable_expressive_conversation=True)`` for the LLM to read
    the turns.

    Example::

        pipeline = Pipeline([transport.input(), stt, PipecatToneReader(), user_aggregator, llm, ...])
    """

    def __init__(
        self,
        *,
        model: PipecatToneReaderModel | None = None,
        precision: Literal["fp16", "fp32"] = "fp16",
        threads: int = 4,
        **kwargs,
    ):
        """Initialize the processor.

        Args:
            model: The model to read with, or None to download Pipecat Tone
                from Hugging Face at start.
            precision: The weights to download: ``"fp16"`` or ``"fp32"``.
            threads: The CPU threads for one read.
            **kwargs: Additional arguments passed to the parent.
        """
        super().__init__(**kwargs)
        self._model = model
        self._precision: Literal["fp16", "fp32"] = precision
        self._threads = threads
        self._previous: str | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Read final transcriptions, and keep the bot's replies to read them against.

        Args:
            frame: The frame to process.
            direction: The direction of frame processing.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            await self._start()
            await self.push_frame(frame, direction)
            await self.push_frame(
                PerceptionMetadataFrame(service_name=self.name, perception=self._capabilities())
            )
            return
        if isinstance(frame, TranscriptionFrame) and frame.text.strip():
            frame.perception = await self._read(frame.text)
        elif isinstance(frame, LLMContextAssistantTurnFrame):
            self._previous = frame.text
        await self.push_frame(frame, direction)

    async def _start(self):
        if self._model is None:
            self._model = await asyncio.to_thread(
                PipecatToneReaderModel.download, precision=self._precision, threads=self._threads
            )

    def _capabilities(self) -> PerceptionCapabilities:
        assert self._model
        return PerceptionCapabilities(
            signals=[
                PerceptionTerm("emotion", _EMOTION, labels=list(self._model.emotions)),
                PerceptionTerm("sarcasm", _SARCASM),
            ]
        )

    async def _read(self, text: str) -> list[PerceptionSignal] | None:
        assert self._model
        try:
            tags = await asyncio.to_thread(self._model.tag, text, self._previous)
        except Exception as e:
            await self.push_error(f"Pipecat Tone couldn't read the turn: {e}", e)
            return None
        likeliest = sorted(tags["emotion"]["probabilities"].items(), key=lambda p: -p[1])[:2]
        return [
            PerceptionSignal("emotion", labels=likeliest),
            PerceptionSignal("sarcasm", value=round(tags["sarcasm"]["probability"], 2)),
        ]
