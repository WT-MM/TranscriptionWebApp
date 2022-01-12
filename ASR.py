import torch
import librosa

from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from speechbrain.pretrained import EncoderDecoderASR

class ASR:
    def __init__(self):
        #SpeechBrain pretrained model
        self.sbmodel = EncoderDecoderASR.from_hparams(
             source="speechbrain/asr-crdnn-rnnlm-librispeech",savedir="pretrained_models/EncoderDecoderASR"
        )

        #Wav2Vec2 processor and model
        self.w2v2processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.w2v2model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        

    def sb(self, file):
        data, rate = librosa.load(file, 16000)  
        data_tensor = torch.tensor(data)
        data_tensor = self.sbmodel.audio_normalizer(data_tensor, 16000)
        batch = data_tensor.unsqueeze(0)
        rel_length = torch.tensor([1.0])
        predicted_words, predicted_tokens = self.sbmodel.transcribe_batch(
            batch, rel_length
        )
        return predicted_words[0]

    def w2v2(self, file):
        audio, rate = librosa.load(file, sr=16000)
        #print(audio)
        input_values = self.w2v2processor(audio, return_tensors="pt", padding="longest", sampling_rate=16000).input_values     
        logits = self.w2v2model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        return self.w2v2processor.batch_decode(predicted_ids)