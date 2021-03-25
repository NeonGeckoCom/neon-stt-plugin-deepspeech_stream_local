# NeonAI Deepspeech STT Plugin
[Mycroft](https://mycroft-ai.gitbook.io/docs/mycroft-technologies/mycroft-core/plugins) compatible
STT Plugin for local Deepspeech streaming Speech-to-Text.

# Configuration:
You may specify your own model and scorer files, otherwise the below default should be used.

```yaml
tts:
    module: deepspeech_stream_local
    deepspeech_stream_local:
      model_path: ~/.local/share/neon/deepspeech-0.9.3-models.pbmm
      scorer_path: ~/.local/share/neon/deepspeech-0.9.3-models.scorer

```