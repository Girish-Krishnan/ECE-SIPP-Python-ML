import whisper


def main():
    model = whisper.load_model('base')
    audio = 'assets/sample.wav'
    result = model.transcribe(audio)
    print('Transcription:')
    print(result['text'])


if __name__ == '__main__':
    main()
