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
import requests


def get_model(ver="0.9.3", tflite=False):
    try:
        if not os.path.isdir(os.path.expanduser("~/.local/share/neon/")):
            os.makedirs(os.path.expanduser("~/.local/share/neon/"))
        if tflite:
            model_url = f'https://github.com/mozilla/DeepSpeech/releases/download/v{ver}/deepspeech-{ver}-models.tflite'
            model_path = os.path.expanduser(f"~/.local/share/neon/deepspeech-{ver}-models.tflite")
        else:
            model_url = f'https://github.com/mozilla/DeepSpeech/releases/download/v{ver}/deepspeech-{ver}-models.pbmm'
            model_path = os.path.expanduser(f"~/.local/share/neon/deepspeech-{ver}-models.pbmm")
        scorer_url = f'https://github.com/mozilla/DeepSpeech/releases/download/v{ver}/deepspeech-{ver}-models.scorer'
        scorer_path = os.path.expanduser(f"~/.local/share/neon/deepspeech-{ver}-models.scorer")

        if not os.path.isfile(model_path):
            print(f"Downloading {model_url}")
            model = requests.get(model_url, allow_redirects=True)
            with open(model_path, "wb") as out:
                out.write(model.content)

        if not os.path.isfile(scorer_path):
            print(f"Downloading {scorer_url}")
            scorer = requests.get(scorer_url, allow_redirects=True)
            with open(scorer_path, "wb") as out:
                out.write(scorer.content)
            print(f"Model Downloaded to {model_path}")
    except Exception as e:
        print(f"Error getting deepspeech models! {e}")
