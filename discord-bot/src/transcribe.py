import sys
# sys.path.append('/opt/homebrew/bin/')
# print(sys.path)
import whisper

def transcribe(audio):

    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(audio)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    # Convert the data type to 'float'
    # mel = mel.float()
    # mel = mel.to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")

    # decode the audio
    options = whisper.DecodingOptions(fp16 = False) #이거 안하면 자꾸 RuntimeError: "slow_conv2d_cpu" not implemented for 'Half' 에러난다
    #options = whisper.DecodingOptions(language=language) #error 나는데
    result = whisper.decode(model, mel, options)

    return result.text

audioFilePath = sys.argv[1]
# language = sys.argv[2]
#audioFilePath = '../outputs/890096759819366440-1711618254963_stereo.wav'#'./890096759819366440-1710161082484.wav'
#print('audioFilePath:', audioFilePath)

# language = 'ko'
#check gpu
model = whisper.load_model("base")
# print(model.device)
# result = model.transcribe(audioFilePath, language)

result = transcribe(audioFilePath)
print(result)
# print(result['text'])