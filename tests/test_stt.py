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

from threading import Event
from neon_utils.file_utils import get_audio_file_stream

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from neon_stt_plugin_deepspeech_stream_local import DeepSpeechLocalStreamingSTT

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
TEST_PATH = os.path.join(ROOT_DIR, "test_audio")


class TestGetSTT(unittest.TestCase):
    def setUp(self) -> None:
        results_event = Event()
        self.stt = DeepSpeechLocalStreamingSTT(results_event)

    def test_get_stt(self):
        for file in os.listdir(TEST_PATH):
            transcription = os.path.splitext(os.path.basename(file))[0].lower()
            stream = get_audio_file_stream(os.path.join(TEST_PATH, file))
            self.stt.stream_start()
            try:
                while True:
                    chunk = stream.read(1024)
                    self.stt.stream_data(chunk)
            except EOFError:
                pass

            result = self.stt.execute(None)
            self.assertIsNotNone(result, f"Error processing: {file}")
            self.assertIn(transcription, result)


if __name__ == '__main__':
    unittest.main()
