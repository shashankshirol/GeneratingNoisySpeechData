import librosa
import soundfile as sf

def main():
    data, sr = librosa.load("female.wav", sr=None)
    mel_spec = librosa.feature.melspectrogram(y=data, sr=sr)
    audio_back = librosa.feature.inverse.mel_to_audio(mel_spec, sr=sr)
    sf.write("recon.wav", audio_back, sr)
    return

if __name__ == "__main__":
    main()
