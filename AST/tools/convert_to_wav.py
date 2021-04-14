import os
import wave
import struct
import io
import argparse
from pydub import AudioSegment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio-dir', type=str, required=True)
    parser.add_argument('--audio-ext', type=str, required=True)
    parser.add_argument('--wav-dir', type=str, required=True)
    args = parser.parse_args()
    
    audio_dir, audio_ext = args.audio_dir, args.audio_ext
    wav_dir = args.wav_dir
    assert os.path.isdir(audio_dir)
    os.makedirs(wav_dir, exist_ok=True)
    for audio in os.listdir(audio_dir):
        if audio.endswith(audio_ext):
            wav_fname = os.path.splitext(os.path.basename(audio))[0] + '.wav'
            wav_path = os.path.join(wav_dir, wav_fname)
            audio_path = os.path.join(audio_dir, audio)
            
            if not os.path.isfile(wav_path) and os.stat(audio_path).st_size > 0:
                print(f'{audio_path} -> {wav_path}')
                sound = AudioSegment.from_file(audio_path)
                sound = sound.set_frame_rate(16000).set_channels(1)
                sound.export(wav_path, format="wav")


if __name__ == "__main__":
    main()