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
import shutil
from inspect import signature

import deepspeech
import numpy as np
import time
import math
from queue import Queue
from neon_stt_plugin_deepspeech_stream_local.util import get_model
from huggingface_hub import hf_hub_download

try:
    from neon_speech.stt import StreamingSTT, StreamThread
except ImportError:
    from ovos_plugin_manager.templates.stt import StreamingSTT, StreamThread
from neon_utils.logger import LOG


class DeepSpeechLocalStreamingSTT(StreamingSTT):
    """
        Streaming STT interface for DeepSpeech
    """

    def __init__(self, results_event, config=None):
        if len(signature(super(DeepSpeechLocalStreamingSTT, self).__init__).parameters) == 0:
            LOG.warning(f"Deprecated Signature Found; config will be ignored and results_event will not be handled!")
            super(DeepSpeechLocalStreamingSTT, self).__init__()
        else:
            super(DeepSpeechLocalStreamingSTT, self).__init__(results_event=results_event, config=config)

        if not hasattr(self, "results_event"):
            self.results_event = None
        # override language with module specific language selection
        self.language = self.config.get('lang') or self.lang
        self.queue = None
        self._clients = dict()
        if self.config.get("model_file") and \
                os.path.isfile(self.config['model_file']):
            try:
                model = deepspeech.Model(self.config.get('model_file'))
                if self.config.get('scorer_file') and \
                        os.path.isfile(self.config['scorer_file']):
                    model.enableExternalScorer(self.config.get('scorer_file'))
                self._clients[self.language.split('-')[0]] = model
            except Exception as e:
                LOG.exception(e)
        self.init_language_model(self.language.split('-')[0], True)
        LOG.debug("Deepspeech STT Ready")

    def create_streaming_thread(self):
        self.queue = Queue()
        return DeepSpeechLocalStreamThread(
            self.queue,
            self.language,
            self,
            self.results_event
        )

    def init_language_model(self, lang: str, cache: bool = True):
        lang = (lang or self.lang).split('-')[0]
        if lang not in self._clients:
            model, scorer = self.download_model(lang)
            LOG.info(f"Loading model for {lang}")
            client = deepspeech.Model(model)
            if scorer and os.path.isfile(scorer):
                LOG.info(f"Enabling scorer for {lang}")
                client.enableExternalScorer(scorer)
            if cache:
                self._clients[lang] = client
            else:
                return client
        return self._clients.get(lang)

    def download_model(self, lang: str = None):
        '''
        Downloading model and scorer for the specific language
        from Huggingface.
        Creating a folder  'polyglot_models' in xdg_data_home
        Creating a language folder in 'polyglot_models' folder
        '''
        lang = lang or self.lang
        repo_id = f"NeonBohdan/stt-polyglot-{lang.split('-')[0]}"
        download_path = hf_hub_download(repo_id, filename="output_graph.pbmm")
        scorer_file_path = hf_hub_download(repo_id, filename="kenlm.scorer")
        # Model path must include the `pbmm` file extension
        # TODO: Consider renaming files and moving to ~/.local/share/neon
        model_path = f"{download_path}.pbmm"
        if not os.path.isfile(model_path) or \
                os.path.getmtime(model_path) != os.path.getmtime(download_path):
            LOG.info("Getting new model from huggingface")
            shutil.copy2(download_path, model_path)
        return model_path, scorer_file_path


class DeepSpeechLocalStreamThread(StreamThread):
    def __init__(self, queue, lang, stt_class, results_event):
        super().__init__(queue, lang)
        self.get_client = stt_class.init_language_model
        self.results_event = results_event
        self.transcriptions = []

        self._invalid_first_transcriptions = ["he"]  # Known bad transcriptions that should be of lower confidence

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

        LOG.info(f"Getting client stream for: {language}")
        stream = self.get_client(language).createStream()
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
        if not self.transcriptions[0]:
            LOG.info("First transcription is empty")
            self.text = None
            self.transcriptions = []
        elif has_data:  # Model sometimes returns transcripts for absolute silence
            if self.transcriptions[0] in self._invalid_first_transcriptions:
                LOG.info(f"Pushing {self.transcriptions[0]} to end of list")
                self.transcriptions.append(self.transcriptions.pop(0))
            LOG.debug("Audio had data")
            self.text = self.transcriptions[0]
        else:
            LOG.warning("Audio was empty")
            self.text = None
            self.transcriptions = []
        if self.results_event:
            self.results_event.set()
        LOG.debug(f"self.text={self.text}")
        return self.transcriptions
