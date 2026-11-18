# Moonshine examples

Please note: since the book was originally published, the moonshine-voice API has undergone some changes. Because of this, what you see in this repository will be different than what you see in the first edition, first printing of the book:

- `transcribe_moonshine.py`: We removed the creation of the `model_path` and `model_arch` variables and we removed all arguments from the `MicTranscriber` constructor (instead using chained methods to configure the `mic_transcriber` object).
- `command_moonshine.py`: Switched from `IntentRecognizer` to `AgentFlow`. `MicTranscriber` is no longer needed here.
- `command_moonshine_tts.py`: Same as `command_moonshine.py` but it's using the moonshine TTS to speak to us.
- `command_moonshine_tts_cache.py`: Same as `command_moonshine_tts.py` but it's using a TTS object and caching its utterances.