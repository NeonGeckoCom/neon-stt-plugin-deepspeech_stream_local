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
import sys
import unittest

from threading import Event
from time import time
from neon_utils.file_utils import get_audio_file_stream

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from neon_stt_plugin_google_cloud_streaming import GoogleCloudStreamingSTT

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
TEST_PATH = os.path.join(ROOT_DIR, "test_audio")


class TestGetSTT(unittest.TestCase):
    def setUp(self) -> None:
        results_event = Event()
        self.stt = GoogleCloudStreamingSTT(results_event)

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

            start_time = time()
            result = self.stt.execute(None)
            exec_time = time() - start_time
            print(exec_time)  # TODO: Report metric
            self.assertEqual(result[0].lower(), transcription)
        # sleep(10)


if __name__ == '__main__':
    unittest.main()
