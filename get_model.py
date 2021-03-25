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
import requests

try:
    if not os.path.isdir(os.path.expanduser("~/.local/share/neon/")):
        os.makedirs(os.path.expanduser("~/.local/share/neon/"))
    model_url = 'https://github.com/mozilla/DeepSpeech/releases/download/v0.8.1/deepspeech-0.8.1-models.pbmm'
    scorer_url = 'https://github.com/mozilla/DeepSpeech/releases/download/v0.8.1/deepspeech-0.8.1-models.scorer'
    model_path = os.path.expanduser("~/.local/share/neon/deepspeech-0.8.1-models.pbmm")
    scorer_path = os.path.expanduser("~/.local/share/neon/deepspeech-0.8.1-models.scorer")

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