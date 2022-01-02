import torch
import librosa

from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from speechbrain.pretrained import EncoderDecoderASR

class ASR:
    def __init__(self):
        #SpeechBrain pretrained model
        tmpdir = getfixture("tmpdir")
        self.sbmodel = EncoderDecoderASR.from_hparams(
            source="speechbrain/asr-crdnn-rnnlm-librispeech",
            savedir=tmpdir,
        )

        #Wav2Vec2 processor and model
        self.w2v2processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.w2v2model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        

    def sb(self, file):
        return self.sbmodel.transcribe_file(file)

    def w2v2(self, file):
        audio, rate = librosa.load(file, sr=16000)
        input_values = self.processor(audio, return_tensors="pt", padding="longest", sampling_rate=16000).input_values     
        logits = self.model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        return self.processor.batch_decode(predicted_ids)