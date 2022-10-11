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
import sys
import unittest
from os.path import isfile

from threading import Event
from neon_utils.file_utils import get_audio_file_stream

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from neon_stt_plugin_deepspeech_stream_local import DeepSpeechLocalStreamingSTT

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
TEST_PATH = os.path.join(ROOT_DIR, "test_audio")


class NeonSTT(DeepSpeechLocalStreamingSTT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def stream_stop(self):
        if self.stream is not None:
            self.queue.put(None)
            text = self.stream.finalize()
            to_return = [text]
            self.stream.join()
            if hasattr(self.stream, 'transcriptions'):
                to_return = self.stream.transcriptions
            self.stream = None
            self.queue = None
            self.results_event.set()
            return to_return
        return None


class TestGetSTT(unittest.TestCase):
    def test_get_stt_simple(self):
        stt = DeepSpeechLocalStreamingSTT()
        for file in os.listdir(TEST_PATH):
            transcription = os.path.splitext(os.path.basename(file))[0].lower()
            stream = get_audio_file_stream(os.path.join(TEST_PATH, file))
            stt.stream_start()
            try:
                while True:
                    chunk = stream.read(1024)
                    stt.stream_data(chunk)
            except EOFError:
                pass

            result = stt.execute(None)
            self.assertIsNotNone(result, f"Error processing: {file}")
            self.assertIsInstance(result, str)
            self.assertEqual(transcription, result)

    def test_get_stt_neon(self):
        results_event = Event()
        stt = NeonSTT(results_event=results_event)
        for file in os.listdir(TEST_PATH):
            transcription = os.path.splitext(os.path.basename(file))[0].lower()
            stream = get_audio_file_stream(os.path.join(TEST_PATH, file))
            stt.stream_start()
            try:
                while True:
                    chunk = stream.read(1024)
                    stt.stream_data(chunk)
            except EOFError:
                pass

            result = stt.execute(None)
            self.assertIsNotNone(result, f"Error processing: {file}")
            self.assertIsInstance(result, list)
            self.assertIn(transcription, result)
            self.assertNotEqual(result[0], 'he')

    def test_available_languages(self):
        stt = DeepSpeechLocalStreamingSTT(None)
        self.assertIsInstance(stt.available_languages, set)
        self.assertIn("en", stt.available_languages)
        self.assertIn("es", stt.available_languages)

    def test_download_model(self):
        stt = DeepSpeechLocalStreamingSTT(None)
        for lang in stt.available_languages:
            model, scorer = stt.download_model(lang, False)
            self.assertTrue(isfile(model))
            self.assertTrue(isfile(scorer))

            tf_model, tf_scorer = stt.download_model(lang, True)
            self.assertTrue(isfile(tf_model))
            self.assertEqual(scorer, tf_scorer)


class TestUtils(unittest.TestCase):
    def test_languages(self):
        from neon_stt_plugin_deepspeech_stream_local.languages import languages
        self.assertIsInstance(languages, dict)
        for lang in languages.keys():
            self.assertEqual(set(languages[lang].keys()),
                             {'repo', 'scorer', 'pbmm', 'tflite', 'language',
                              'regions'})
            for region in languages[lang]['regions']:
                self.assertIsInstance(region, tuple)
                self.assertIsInstance(region[0], str)
                self.assertEqual(len(region[1].split('-')), 2)

    def test_stt_config(self):
        from neon_stt_plugin_deepspeech_stream_local.languages import stt_config
        self.assertIsInstance(stt_config, dict)
        for lang, configs in stt_config.items():
            self.assertIsInstance(configs, list)
            for config in configs:
                self.assertEqual(config['lang'], lang)
                self.assertIsInstance(config['display_name'], str)
                self.assertTrue(config['offline'])


if __name__ == '__main__':
    unittest.main()
