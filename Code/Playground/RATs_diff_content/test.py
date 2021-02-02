from pydub import AudioSegment

audio = AudioSegment.from_wav("output\start_scale=3_reconstructed.wav")

quiter_audio = audio - 10

quiter_audio.export("quiter_10.wav", "wav")