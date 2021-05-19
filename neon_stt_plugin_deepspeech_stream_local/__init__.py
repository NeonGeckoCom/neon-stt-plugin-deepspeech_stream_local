#!/usr/bin/env bash

# NEON AI (TM) SOFTWARE, Software Development Kit & Application Development System
# All trademark and other rights reserved by their respective owners
# Copyright 2008-2021 Neongecko.com Inc.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
# following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions
#    and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions
#    and the following disclaimer in the documentation and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote
#    products derived from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
# USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
from inspect import signature

import deepspeech
import numpy as np
import time
import math
from queue import Queue
try:
    from neon_speech.stt import StreamingSTT, StreamThread
except ImportError:
    from ovos_plugin_manager.templates.stt import StreamingSTT, StreamThread
from neon_utils.logger import LOG
LOG.name = "stt_plugin"


class DeepSpeechLocalStreamingSTT(StreamingSTT):
    """
        Streaming STT interface for DeepSpeech
    """

    def __init__(self, results_event, config=None):
        if len(signature(super(DeepSpeechLocalStreamingSTT, self).__init__).parameters) == 2:
            super(DeepSpeechLocalStreamingSTT, self).__init__(results_event, config)
        else:
            LOG.warning(f"Shorter Signature Found; config will be ignored and results_event will not be handled!")
            super(DeepSpeechLocalStreamingSTT, self).__init__()
            self.results_event = None
        # override language with module specific language selection
        self.language = self.config.get('lang') or self.lang
        self.queue = None
        if not self.language.startswith("en"):
            raise ValueError("DeepSpeech is currently english only")

        model_path = self.config.get("model_path") or \
            os.path.expanduser("~/.local/share/neon/deepspeech-0.9.3-models.pbmm")
        scorer_path = self.config.get("scorer_path") or \
            os.path.expanduser("~/.local/share/neon/deepspeech-0.9.3-models.scorer")
        if not os.path.isfile(model_path):
            LOG.error("You need to provide a valid model file")
            LOG.error(model_path)
            LOG.info("download a model from https://github.com/mozilla/DeepSpeech")
            raise FileNotFoundError
        if not scorer_path or not os.path.isfile(scorer_path):
            LOG.warning("You should provide a valid scorer")
            LOG.info("download scorer from https://github.com/mozilla/DeepSpeech")

        self.client = deepspeech.Model(model_path)
        if scorer_path:
            self.client.enableExternalScorer(scorer_path)

    def create_streaming_thread(self):
        self.queue = Queue()
        return DeepSpeechLocalStreamThread(
            self.queue,
            self.language,
            self.client,
            self.results_event
        )


class DeepSpeechLocalStreamThread(StreamThread):
    def __init__(self, queue, lang, client, results_event):
        super().__init__(queue, lang)
        self.client = client
        self.results_event = results_event
        self.transcriptions = []

    def handle_audio_stream(self, audio, language):
        short_normalize = (1.0 / 32768.0)
        swidth = 2
        threshold = 10
        timeout_length = 5

        def rms(frame):
            count = len(frame) / swidth
            sum_squares = 0.0
            for sample in frame:
                n = sample * short_normalize
                sum_squares += n * n
            rms_value = math.pow(sum_squares / count, 0.5)
            return rms_value * 1000

        stream = self.client.createStream()
        current_time = time.time()
        end_time = current_time + timeout_length
        previous_intermediate_result, current_intermediate_result = '', ''
        has_data = False
        for data in audio:
            data16 = np.frombuffer(data, dtype=np.int16)
            if data16.max() != data16.min():
                has_data = True
            current_time = time.time()
            stream.feedAudioContent(data16)
            current_intermediate_result = stream.intermediateDecode()
            if rms(data16) > threshold and current_intermediate_result != previous_intermediate_result:
                end_time = current_time + timeout_length
            previous_intermediate_result = current_intermediate_result
            if current_time > end_time:
                break
        responses = stream.finishStreamWithMetadata(num_results=5)
        self.transcriptions = []
        # LOG.debug(f"The responses are {responses}")
        for transcript in responses.transcripts:
            letters = [token.text for token in transcript.tokens]
            self.transcriptions.append("".join(letters).strip())
        # self.transcriptions = responses
        LOG.debug(self.transcriptions)
        if has_data:  # Model sometimes returns transcripts for absolute silence
            LOG.debug(f"Audio had data!!")
            self.text = self.transcriptions[0]
        else:
            LOG.warning(f"Audio was empty!")
            self.text = None
            self.transcriptions = []
        if self.results_event:
            self.results_event.set()
        return self.transcriptions
