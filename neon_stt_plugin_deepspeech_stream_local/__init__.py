#!/usr/bin/env bash

# NEON AI (TM) SOFTWARE, Software Development Kit & Application Development System
#
# Copyright 2008-2020 Neongecko.com Inc. | All Rights Reserved
#
# Notice of License - Duplicating this Notice of License near the start of any file containing
# a derivative of this software is a condition of license for this software.
# Friendly Licensing:
# No charge, open source royalty free use of the Neon AI software source and object is offered for
# educational users, noncommercial enthusiasts, Public Benefit Corporations (and LLCs) and
# Social Purpose Corporations (and LLCs). Developers can contact developers@neon.ai
# For commercial licensing, distribution of derivative works or redistribution please contact licenses@neon.ai
# Distributed on an "AS IS” basis without warranties or conditions of any kind, either express or implied.
# Trademarks of Neongecko: Neon AI(TM), Neon Assist (TM), Neon Communicator(TM), Klat(TM)
# Authors: Guy Daniels, Daniel McKnight, Regina Bloomstine, Elon Gasper, Richard Leeds
#
# Specialized conversational reconveyance options from Conversation Processing Intelligence Corp.
# US Patents 2008-2020: US7424516, US20140161250, US20140177813, US8638908, US8068604, US8553852, US10530923, US10530924
# China Patent: CN102017585  -  Europe Patent: EU2156652  -  Patents Pending

import os
import deepspeech
import numpy as np
import time
import math

from queue import Queue

from mycroft.stt import StreamingSTT, StreamThread
from mycroft.util.log import LOG


class DeepSpeechLocalStreamThread(StreamThread):
    def __init__(self, queue, lang, client, results_event):
        super().__init__(queue, lang)
        self.client = client
        self.results_event = results_event

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
        responses = stream.finishStream()
        LOG.debug(f"The responses are {responses}")
        self.transcriptions = responses

        if has_data:  # Model sometimes returns transcripts for absolute silence
            LOG.debug(f"Audio had data!!")
        else:
            LOG.warning(f"Audio was empty!")
            self.transcriptions = None
        self.results_event.set()
        return self.transcriptions


class DeepSpeechLocalStreamingSTT(StreamingSTT):
    """
        Streaming STT interface for DeepSpeech
    """

    def __init__(self, results_event):
        super(DeepSpeechLocalStreamingSTT, self).__init__(results_event)
        # override language with module specific language selection
        self.language = self.config.get('lang') or self.lang
        if not self.language.startswith("en"):
            raise ValueError("DeepSpeech is currently english only")

        model_path = self.config.get("model_path",
                                     os.path.expanduser("~/.local/share/neon/deepspeech-0.8.1-models.pbmm"))
        scorer_path = self.config.get("scorer_path",
                                      os.path.expanduser("~/.local/share/neon/deepspeech-0.8.1-models.scorer"))
        if not os.path.isfile(model_path):
            LOG.error("You need to provide a valid model file")
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
