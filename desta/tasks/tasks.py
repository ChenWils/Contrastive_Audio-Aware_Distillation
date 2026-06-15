import json
from tqdm import tqdm
import numpy as np
import torch
import os
from typing import List, Dict, Optional, Union
from torch.utils.data import DataLoader

def split_into_batches(lst, batch_size):
    return [lst[i:i + batch_size] for i in range(0, len(lst), batch_size)]


# # # # # # # # # # # # # # # # # # # #
# Automatic Speech Recognition (ASR)
# # # # # # # # # # # # # # # # # # # #

def run_asr(args,
            task_name,
            input_data_dict,
            remaining_data,
            output_manifest_path,
            logger,
            model_id="openai/whisper-large-v3"):
    """
    pip install transformers
    """

    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline
    from datasets import Dataset
    from tqdm import tqdm
    import json

    logger.info(f"Model ID: {model_id}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = WhisperForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True,
    )
    model.to(device)
    processor = WhisperProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=256,
        chunk_length_s=30,
        torch_dtype=torch_dtype,
        device=device,
    )

    filelist = [{"path": item["audio_filepath"]} for item in remaining_data]
    dataset = Dataset.from_list(filelist)

    def iterate_data(dataset):
        for i, item in enumerate(dataset):
            yield item["path"]
    with open(output_manifest_path, "a") as fo:
        for path, out in tqdm(zip(iterate_data(dataset), pipe(iterate_data(dataset), batch_size=args.batch_size, return_timestamps=True, generate_kwargs={"task": "transcribe"}))):
            data = input_data_dict[path]
            data[task_name] = {
                "text": out["text"],
                "chunk": out["chunks"],
            }
            fo.write(json.dumps(data, ensure_ascii=False) + "\n")

    model.cpu()
    del model



# # # # # # # # # # # # # # # # # # # #
# Gender Recognition                   
# # # # # # # # # # # # # # # # # # # #

def run_gender_recognition(args, 
                           task_name,
                           input_data_dict,
                           remaining_data,
                           output_manifest_path,
                           logger,
                           model_id="alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech"):
    import torch
    import torchaudio
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, Wav2Vec2Processor
    from torch.utils.data import DataLoader
    from torch.nn import functional as F
    from tqdm import tqdm
    import json
    import numpy as np
    import os

    logger.info(f"Model ID: {model_id}")

    class GenderRecognitionDataset(torch.utils.data.Dataset):
        def __init__(
            self,
            dataset: List,
            basedir: Optional[str] = None,
            sampling_rate: int = 16000,
            max_audio_len: int = 5,
        ):
            self.dataset = dataset
            self.basedir = basedir

            self.sampling_rate = sampling_rate
            self.max_audio_len = max_audio_len

        def __len__(self):
            """
            Return the length of the dataset
            """
            return len(self.dataset)

        def __getitem__(self, index):
            if self.basedir is None:
                filepath = self.dataset[index]
            else:
                filepath = os.path.join(self.basedir, self.dataset[index])

            speech_array, sr = torchaudio.load(filepath)

            if speech_array.shape[0] > 1:
                speech_array = torch.mean(speech_array, dim=0, keepdim=True)

            if sr != self.sampling_rate:
                transform = torchaudio.transforms.Resample(sr, self.sampling_rate)
                speech_array = transform(speech_array)
                sr = self.sampling_rate
            speech_array = speech_array.squeeze().numpy()
            return {"input_values": speech_array, "attention_mask": None, "audio_path": filepath}

    class CollateFunc:
        def __init__(
            self,
            processor: Wav2Vec2Processor,
            padding: Union[bool, str] = True,
            pad_to_multiple_of: Optional[int] = None,
            return_attention_mask: bool = True,
            sampling_rate: int = 16000,
            max_length: Optional[int] = None,
        ):
            self.sampling_rate = sampling_rate
            self.processor = processor
            self.padding = padding
            self.pad_to_multiple_of = pad_to_multiple_of
            self.return_attention_mask = return_attention_mask
            self.max_length = max_length

        def __call__(self, batch: List[Dict[str, np.ndarray]]):
            # Extract input_values from the batch
            input_values = [item["input_values"] for item in batch]
            audio_paths = [item["audio_path"] for item in batch]
            batch = self.processor(
                input_values,
                sampling_rate=self.sampling_rate,
                return_tensors="pt",
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_attention_mask=self.return_attention_mask
            )

            return {
                "input_values": batch.input_values,
                "attention_mask": batch.attention_mask if self.return_attention_mask else None,
                "audio_path": audio_paths
            }
    
    # Gender recognition specific code
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
    model = AutoModelForAudioClassification.from_pretrained(
        pretrained_model_name_or_path=model_id,
        num_labels=2,
        label2id={"female": 0, "male": 1},
        id2label={0: "female", 1: "male"},
    )

    test_dataset = GenderRecognitionDataset([item["audio_filepath"] for item in remaining_data], max_audio_len=30)
    data_collator = CollateFunc(
        processor=feature_extractor,
        padding=True,
        sampling_rate=16000,
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        collate_fn=data_collator,
        shuffle=False,
        num_workers=2
    )
    
    model.to("cuda")
    model.eval()

    with torch.no_grad():
        with open(output_manifest_path, "a") as fo:
            for batch in tqdm(test_dataloader):
                input_values, attention_mask = batch['input_values'].to("cuda"), batch['attention_mask'].to("cuda")

                logits = model(input_values, attention_mask=attention_mask).logits
                scores = F.softmax(logits, dim=-1)

                pred = torch.argmax(scores, dim=1).cpu().detach().numpy()
                score = torch.max(scores, dim=1).values.cpu()

                for audio_path, p, s in zip(batch["audio_path"], pred, score):
                
                    data = input_data_dict[audio_path]
                    data[task_name] = {
                        "text": {0: "female", 1: "male"}[p].capitalize(),
                        "raw": s.item()
                    }
                    fo.write(json.dumps(data) + "\n")

    del model

# # # # # # # # # # # # # # # # # # # #
# Gender age
# # # # # # # # # # # # # # # # # # # #

def run_gender_age(args, 
                   task_name,
                   input_data_dict,
                   remaining_data,
                   output_manifest_path,
                   logger,
                   model_id='audeering/wav2vec2-large-robust-24-ft-age-gender'):
    import numpy as np
    import torch
    import torch.nn as nn
    from transformers import Wav2Vec2Processor
    from transformers.models.wav2vec2.modeling_wav2vec2 import (
        Wav2Vec2Model,
        Wav2Vec2PreTrainedModel,
    )
    from torch.utils.data import DataLoader, Dataset
    from tqdm import tqdm
    import torchaudio

    class GenderRecognitionDataset(torch.utils.data.Dataset):
        def __init__(
            self,
            dataset: List,
            basedir: Optional[str] = None,
            sampling_rate: int = 16000,
            max_audio_len: int = 5,
        ):
            self.dataset = dataset
            self.basedir = basedir

            self.sampling_rate = sampling_rate
            self.max_audio_len = max_audio_len

        def __len__(self):
            """
            Return the length of the dataset
            """
            return len(self.dataset)

        def __getitem__(self, index):
            if self.basedir is None:
                filepath = self.dataset[index]
            else:
                filepath = os.path.join(self.basedir, self.dataset[index])

            speech_array, sr = torchaudio.load(filepath)

            if speech_array.shape[0] > 1:
                speech_array = torch.mean(speech_array, dim=0, keepdim=True)

            if sr != self.sampling_rate:
                transform = torchaudio.transforms.Resample(sr, self.sampling_rate)
                speech_array = transform(speech_array)
                sr = self.sampling_rate
            speech_array = speech_array.squeeze().numpy()
            return {"input_values": speech_array, "attention_mask": None, "audio_path": filepath}

    class CollateFunc:
        def __init__(
            self,
            processor: Wav2Vec2Processor,
            padding: Union[bool, str] = True,
            pad_to_multiple_of: Optional[int] = None,
            return_attention_mask: bool = True,
            sampling_rate: int = 16000,
            max_length: Optional[int] = None,
        ):
            self.sampling_rate = sampling_rate
            self.processor = processor
            self.padding = padding
            self.pad_to_multiple_of = pad_to_multiple_of
            self.return_attention_mask = return_attention_mask
            self.max_length = max_length

        def __call__(self, batch: List[Dict[str, np.ndarray]]):
            # Extract input_values from the batch
            input_values = [item["input_values"] for item in batch]
            audio_paths = [item["audio_path"] for item in batch]
            batch = self.processor(
                input_values,
                sampling_rate=self.sampling_rate,
                return_tensors="pt",
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_attention_mask=self.return_attention_mask
            )

            return {
                "input_values": batch.input_values,
                "attention_mask": batch.attention_mask if self.return_attention_mask else None,
                "audio_path": audio_paths
            }
            
    class ModelHead(nn.Module):
        r"""Classification head."""

        def __init__(self, config, num_labels):

            super().__init__()

            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
            self.dropout = nn.Dropout(config.final_dropout)
            self.out_proj = nn.Linear(config.hidden_size, num_labels)

        def forward(self, features, **kwargs):

            x = features
            x = self.dropout(x)
            x = self.dense(x)
            x = torch.tanh(x)
            x = self.dropout(x)
            x = self.out_proj(x)

            return x


    class AgeGenderModel(Wav2Vec2PreTrainedModel):
        r"""Speech emotion classifier."""

        def __init__(self, config):

            super().__init__(config)

            self.config = config
            self.wav2vec2 = Wav2Vec2Model(config)
            self.age = ModelHead(config, 1)
            self.gender = ModelHead(config, 3)
            self.init_weights()

        def forward(
                self,
                input_values,
        ):

            outputs = self.wav2vec2(input_values)
            hidden_states = outputs[0]
            hidden_states = torch.mean(hidden_states, dim=1)
            logits_age = self.age(hidden_states)
            logits_gender = torch.softmax(self.gender(hidden_states), dim=1)

            return hidden_states, logits_age, logits_gender

    model = AgeGenderModel.from_pretrained(model_id)
    processor = Wav2Vec2Processor.from_pretrained(model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    test_dataset = GenderRecognitionDataset([item["audio_filepath"] for item in remaining_data], max_audio_len=30)
    data_collator = CollateFunc(
        processor=processor,
        padding=True,
        sampling_rate=16000,
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        collate_fn=data_collator,
        shuffle=False,
        num_workers=2
    )
    
    model.to("cuda")
    model.eval()

    with torch.no_grad():
        with open(output_manifest_path, "a") as fo:
            for batch in tqdm(test_dataloader):
                input_values, attention_mask = batch['input_values'].to("cuda"), batch['attention_mask'].to("cuda")
                _, logits_age, logits_gender = model(input_values)

                for audio_path, age, gender in zip(batch["audio_path"], logits_age, logits_gender):
                
                    data = input_data_dict[audio_path]
                    data["age"] = {
                        "text": round(age.item()*100, -1), # round to nearest 10
                        "raw": age.item()*100
                    }
                    idx2label = {0: "Female", 1: "Male", 2: "Child"}
                    
                    data["gender"] = {
                        "text": idx2label[torch.argmax(gender).item()],
                        "raw": gender.cpu().numpy().tolist()[torch.argmax(gender).item()]
                    }
                    fo.write(json.dumps(data) + "\n")
        
    del model



# # # # # # # # # # # # # # # # # # # #
# Emotion Recognition                   
# # # # # # # # # # # # # # # # # # # #

def run_emotion_recognition(args, 
                            task_name,
                            input_data_dict,
                            remaining_data,
                            output_manifest_path,
                            logger,
                            model_id="iic/emotion2vec_plus_large"):
    """
    pip install -U funasr modelscope
    """

    from modelscope.pipelines import pipeline as modelscope_pipeline
    from modelscope.utils.constant import Tasks
    from tqdm import tqdm
    import json

    logger.info(f"Model ID: {model_id}")

    # Emotion recognition specific code
    # inference_pipeline = modelscope_pipeline(
    #     task=Tasks.emotion_recognition,
    #     model=model_id
    # )

    # def emotion_recognition(audio_path):
    #     rec_result = inference_pipeline(audio_path)
    #     data = rec_result[0]
    #     mask_labels = ['其他/other', '<unk>']
    #     masked_scores = [(label, score) for label, score in zip(data['labels'], data['scores']) if label not in mask_labels]

    #     highest_label, highest_score = max(masked_scores, key=lambda x: x[1])
    #     highest_label = highest_label.split('/')[-1]

    #     return highest_label, highest_score
    


    # with open(output_manifest_path, "a") as fo:
    #     for item in tqdm(remaining_data):
    #         audio_path = item["audio_filepath"]
    #         highest_label, highest_score = emotion_recognition(audio_path)
    #         item[task_name] = {
    #             "text": highest_label,
    #             "raw": highest_score
    #         }
        
    #         fo.write(json.dumps(item, ensure_ascii=False) + "\n")

    # del inference_pipeline

    from funasr import AutoModel
    model = AutoModel(model="iic/emotion2vec_plus_large")

    batches = split_into_batches(remaining_data, args.batch_size)
    with open(output_manifest_path, "a") as fo:
        for batch in tqdm(batches):        
            results = model.generate([item["audio_filepath"] for item in batch], output_dir="./outputs", granularity="utterance", extract_embedding=False)
        
            for item, result in zip(batch, results):
                data = result
                mask_labels = ['其他/other', '<unk>']
                masked_scores = [(label, score) for label, score in zip(data['labels'], data['scores']) if label not in mask_labels]

                highest_label, highest_score = max(masked_scores, key=lambda x: x[1])
                highest_label = highest_label.split('/')[-1]

                item[task_name] = {
                    "text": highest_label,
                    "raw": highest_score
                }
            
                fo.write(json.dumps(item, ensure_ascii=False) + "\n")
                
    del model

# # # # # # # # # # # # # # # # # # # #
# snr_c50
# # # # # # # # # # # # # # # # # # # #


def run_snr_c50_estimation(args,
                       task_name,
                       input_data_dict,
                       remaining_data,
                       output_manifest_path,
                       logger,
                       model_id="pyannote/brouhaha"):
    """
    pip install pyannote-audio
    pip install https://github.com/marianne-m/brouhaha-vad/archive/main.zip
    """
    logger.info(f"Model ID: {model_id}")
    
    from pyannote.audio import Model
    from tqdm import tqdm
    import numpy as np
    import os
    import json
    from pyannote.audio import Inference

    model = Model.from_pretrained(model_id)
    inference = Inference(model)
    from brouhaha.pipeline import RegressiveActivityDetectionPipeline

    pipeline = RegressiveActivityDetectionPipeline(segmentation=model)

    
    def round_to_nearest_5(number):
        return round(number / 5) * 5
    with open(output_manifest_path, "a") as fo:
        for item in tqdm(remaining_data):
            audio_path = item["audio_filepath"]
            
            output = pipeline(audio_path)
            
            item["_snr"] = {
                "text": round_to_nearest_5(output["snr"].mean().item()),
                "raw": output["snr"].mean().item(),
            }
            item["_c50"] = {
                "text": round_to_nearest_5(output["c50"].mean().item()),
                "raw": output["c50"].mean().item()
            }    
            fo.write(json.dumps(item) + "\n")

    del model



# # # # # # # # # # # # # # # # # # # #
# Duration
# # # # # # # # # # # # # # # # # # # #

def run_duration(args,
                 task_name,
                 input_data_dict,
                 remaining_data,
                 output_manifest_path,
                 logger,
                ):
    import librosa
    from tqdm import tqdm
    import json

    with open(output_manifest_path, "a") as fo:
        for data in tqdm(remaining_data):
            audio_path = data["audio_filepath"]
            y, sr = librosa.load(audio_path, sr=None)
            data[task_name] = {
                "text": round(len(y) / sr),
                "raw": len(y) / sr
            }

            fo.write(json.dumps(data) + "\n")


# # # # # # # # # # # # # # # # # # # #
# VQscore
# # # # # # # # # # # # # # # # # # # #

def run_vqscore(args,
                task_name,
                input_data_dict,
                remaining_data,
                output_manifest_path,
                logger,
                ):
    from desta.tasks.VQscore.VQVAE_models import VQVAE_SE, VQVAE_QE
    from desta.tasks.VQscore.VQVAE_models import stft_magnitude, cos_loss
    import yaml
    import torchaudio
    import torch
    import numpy as np


    with open(os.path.join(os.path.dirname(__file__), "VQscore", "config.yaml"), "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True

    with torch.no_grad():
        with open(output_manifest_path, "a") as fo:
            if config["task"] == "Quality_Estimation":
                hop_size = 256
                VQVAE = VQVAE_QE(**config['VQVAE_params']).to(device).eval()
                VQVAE.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), "VQscore", "checkpoint-dnsmos_ovr_CC=0.835.pkl"))['model']['VQVAE'])
                
                for data in tqdm(remaining_data):
                    audio_filepath = data["audio_filepath"]
                    
                    speech, fs   = torchaudio.load(audio_filepath)
                    if fs != 16000:
                        speech = torchaudio.functional.resample(speech, fs, 16000).to(device)
                    
                    
                    SP_original = stft_magnitude(speech, hop_size=hop_size)
                    if config['input_transform'] == 'log1p':
                        SP_original = torch.log1p(SP_original)
                    
                    z = VQVAE.CNN_1D_encoder(SP_original.cuda())
                    zq, indices, vqloss, distance = VQVAE.quantizer(z, stochastic=False, update=False)
                    #SP_output = VQVAE.CNN_1D_decoder(zq)                                                                 
                    VQScore_cos_z_original = -cos_loss(z.transpose(2, 1).cpu(), zq.cpu()).numpy()
                    
                    data[task_name] = {
                        "text": VQScore_cos_z_original.item(),
                        "raw": VQScore_cos_z_original.item()
                    }
                
                    fo.write(json.dumps(data) + "\n")
    del VQVAE

# # # # # # # # # # # # # # # # # # # #
# Pitch
# # # # # # # # # # # # # # # # # # # #
def run_pitch(args,
             task_name,
             input_data_dict,
             remaining_data,
             output_manifest_path,
             logger,
             ):
    import penn
    # from dataspeech
    hopsize = .01
    fmin = 30.
    fmax = 1000.
    checkpoint = None
    center = 'half-hop'
    interp_unvoiced_at = .065
    with open(output_manifest_path, "a") as fo:
        for data in tqdm(remaining_data):
            audio_path = data["audio_filepath"]
            
            pitch, periodicity = penn.from_file(
                file=audio_path,
                fmin=fmin,
                fmax=fmax,
                hopsize=hopsize,
                center=center,
                interp_unvoiced_at=interp_unvoiced_at,
                checkpoint=checkpoint,
                gpu=0
            )
            data[task_name+"_mean"] = {
                "text": pitch.mean().cpu().item(),
                "raw": pitch.mean().cpu().item(),
            }
            data[task_name+"_std"] = {
                "text": pitch.std().cpu().item(),
                "raw": pitch.std().cpu().item(),
            }

            fo.write(json.dumps(data) + "\n")


from transformers import Wav2Vec2Processor
import torchaudio
class AudioDataset(torch.utils.data.Dataset):
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    sampling_rate = 16000
    max_length = 30
    return_attention_mask = None


    def __init__(
        self,
        dataset: List,
        basedir: Optional[str] = None,
        sampling_rate: int = 16000,
        max_audio_len: int = 5,
    ):
        self.dataset = dataset
        self.basedir = basedir

        self.sampling_rate = sampling_rate
        self.max_audio_len = max_audio_len
        self.max_length = 30

    def __len__(self):
        """
        Return the length of the dataset
        """
        return len(self.dataset)

    def __getitem__(self, index):
        if self.basedir is None:
            filepath = self.dataset[index]
        else:
            filepath = os.path.join(self.basedir, self.dataset[index])

        speech_array, sr = torchaudio.load(filepath)

        if speech_array.shape[0] > 1:
            speech_array = torch.mean(speech_array, dim=0, keepdim=True)

        if sr != self.sampling_rate:
            transform = torchaudio.transforms.Resample(sr, self.sampling_rate)
            speech_array = transform(speech_array)
            sr = self.sampling_rate
        speech_array = speech_array.squeeze().numpy()
        return {"input_values": speech_array, "wav_lens": speech_array.shape[0] ,"attention_mask": None, "audio_path": filepath}

    @classmethod
    def collate_fn(cls, batch: List[Dict[str, np.ndarray]]):
        # Extract input_values from the batch
        input_values = [item["input_values"] for item in batch]
        audio_paths = [item["audio_path"] for item in batch]
        
        # Process the batch using the processor
        processed_batch = cls.processor(
            input_values,
            sampling_rate=cls.sampling_rate,
            return_tensors="pt",
            padding=True,
            max_length=cls.max_length,
        )

        return {
            "input_values": processed_batch.input_values,
            "attention_mask": processed_batch.attention_mask if cls.return_attention_mask else None,
            "audio_path": audio_paths,
            "wav_lens": torch.tensor([item["wav_lens"] for item in batch])
        }

def run_accent_classfication(args,
             task_name,
             input_data_dict,
             remaining_data,
             output_manifest_path,
             logger,
             ):
    
    from speechbrain.pretrained.interfaces import foreign_class
    classifier = foreign_class(
        source="Jzuluaga/accent-id-commonaccent_xlsr-en-english", pymodule_file="custom_interface.py", classname="CustomEncoderWav2vec2Classifier",
        run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"}
        )

    dataset = AudioDataset([item["audio_filepath"] for item in remaining_data], max_audio_len=30)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        collate_fn=AudioDataset.collate_fn,
        shuffle=False,
        num_workers=2
    )

    with open(output_manifest_path, "a") as fo:
        for data in tqdm(dataloader):
            input_values, wav_lens = data['input_values'].to("cuda"), data['wav_lens'].to("cuda")
            out_prob, score, index, text_lab = classifier.classify_batch(wavs=input_values, wav_lens=wav_lens)
        
            for score, text_lab, audio_filepath in zip(score, text_lab, data["audio_path"]):
                data = input_data_dict[audio_filepath]
                data[task_name] = {
                    "text": text_lab,
                    "raw": score.item()
                }
                fo.write(json.dumps(data) + "\n")

    del classifier
