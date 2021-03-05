# NEON AI (TM) SOFTWARE, Software Development Kit & Application Development System
#
# Copyright 2008-2021 Neongecko.com Inc. | All Rights Reserved
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
# US Patents 2008-2021: US7424516, US20140161250, US20140177813, US8638908, US8068604, US8553852, US10530923, US10530924
# China Patent: CN102017585  -  Europe Patent: EU2156652  -  Patents Pending

import os
import requests
from setuptools import setup

PLUGIN_ENTRY_POINT = 'deepspeech_local_strem = neon_stt_plugin_deepspeech:DeepspeechLocalStreamingSTT'

with open("README.md", "r") as f:
    long_description = f.read()

with open("./version.py", "r", encoding="utf-8") as v:
    for line in v.readlines():
        if line.startswith("__version__"):
            if '"' in line:
                version = line.split('"')[1]
            else:
                version = line.split("'")[1]

with open("./requirements.txt", "r", encoding="utf-8") as r:
    requirements = r.readlines()

try:
    if not os.path.isdir(os.path.expanduser("~/.local/share/neon/")):
        os.makedirs(os.path.expanduser("~/.local/share/neon/"))
    model_url = 'https://github.com/mozilla/DeepSpeech/releases/download/v0.8.1/deepspeech-0.8.1-models.pbmm'
    scorer_url = 'https://github.com/mozilla/DeepSpeech/releases/download/v0.8.1/deepspeech-0.8.1-models.scorer'
    model_path = os.path.expanduser("~/.local/share/neon/deepspeech-0.8.1-models.pbmm")
    scorer_path = os.path.expanduser("~/.local/share/neon/deepspeech-0.8.1-models.scorer")

    model = requests.get(model_url, allow_redirects=True)
    with open(model_path, "wb") as out:
        out.write(model.content)

    scorer = requests.get(scorer_url, allow_redirects=True)
    with open(scorer_path, "wb") as out:
        out.write(scorer.content)
except Exception as e:
    print(f"Error getting deepspeech models! {e}")

setup(
    name='neon-stt-plugin-deepspeech',
    version=version,
    description='A Deepspeech Streaming stt plugin for Neon',
    url='https://github.com/NeonGeckoCom/neon-stt-plugin-deepspeech',
    author='Neongecko',
    author_email='developers@neon.ai',
    license='NeonAI License v1.0',
    packages=['neon_stt_plugin_deepspeech'],
    install_requires=requirements,
    zip_safe=True,
    classifiers=[
        'Intended Audience :: Developers',
        'Topic :: Text Processing :: Linguistic',
        'Programming Language :: Python :: 3.6',
    ],
    keywords='mycroft plugin stt',
    entry_points={'mycroft.plugin.stt': PLUGIN_ENTRY_POINT}
)
