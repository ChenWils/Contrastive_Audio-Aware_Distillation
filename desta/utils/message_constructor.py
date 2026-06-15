import random
from transformers import AutoTokenizer
from copy import deepcopy
from typing import List
from collections import defaultdict
from lulutils import print
class MessagesConstructor:
    def __init__(self, config, tokenizer: AutoTokenizer, input_data: List[dict]):
        self.tokenizer = tokenizer
        self.config = config
        self.input_data = input_data
        
        random.seed(self.config.get("seed"))

        self._prepare_sample_strategy()

    def construct_messages(self, 
                           datas: List[dict], 
                           audio_locator: str = "<|AUDIO|>",
                           ) -> str:
        """
        template is a "messages" chat history that has placeholders for prossesing.

        template = [
            {"role": "system", "content": "{system_prompt}"},
            {"role": "user", "content": "{instruction} <|AUDIO|>"},
        ]
        """
        if self.config.get("system_prompts"):
            system_prompt = random.choice(self.config.get("system_prompts"))
        else:
            raise ValueError("No system prompt found in config")

        if datas[0].get("instruction") is not None:
            instruction = datas[0]["instruction"]
        elif datas[0].get("input") is not None:
            instruction = datas[0]["input"]
        elif self.config.get("instructions") is not None:
            instruction = random.choice(self.config["instructions"])
        else:
            raise ValueError("No instruction or input found in data")
        
        if datas[0].get("messages") is not None:
            template = deepcopy(datas[0]["messages"])
        elif self.config.get("templates") is not None:
            template = deepcopy(random.choice(self.config.get("templates")))
        else:
            raise ValueError("No template messages found in data")

        seed_transcripts = [deepcopy(item["seed_transcript"]) for item in datas]

        new_messages = [] # replace <|AUDIO|> with seed_transcripts
        training_messages = [] # <|AUDIO|> messages for training
        audio_count = 0
        for message in template:
            message["content"] = message["content"].replace("{system_prompt}", system_prompt)
            message["content"] = message["content"].replace("{instruction}", instruction)

            training_messages.append(deepcopy(message))

            for _ in range(message["content"].count(audio_locator)):
                seed_transcript = seed_transcripts.pop(0)
                message["content"] = message["content"].replace(audio_locator, seed_transcript, 1)
                audio_count += 1

            new_messages.append(message)

        # assert audio_count == len(datas), f"audio_count: {audio_count}, len(datas): {len(datas)}"


        prompt = self.tokenizer.apply_chat_template(
                new_messages, tokenize=False, add_generation_prompt=True
            )
        return prompt, training_messages


    
    def construct_training_sample(self, datas: List[dict], messages: List[dict], prompt: str, response: str):
        
        audios = [{
            "audio_filepath": item["audio_filepath"],
            "transcription": self.get_transcription(item),
            "seed_transcript": item["seed_transcript"],
            "duration": item.get("duration", None),
        } for item in datas]

        audio_count = 0
        for message in messages:
            audio_count += message["content"].count("<|AUDIO|>")

        audios = audios[:audio_count]
        training_sample = {
            "id": "@".join([item["audio_filepath"] for item in audios]),
            "audios": audios,
            "messages": messages,
            "prompt": prompt,
            "response": response,
        }

        return training_sample


    def get_transcription(self, data):
        if data.get("text") is not None:
            transcription = data.get("text")
        elif data.get("transcription") is not None:
            transcription = data.get("transcription")
        elif data.get("transcript") is not None:
            transcription = data.get("transcript")
        elif data.get("_asr") is not None:
            transcription = data["_asr"]["text"]
        else:
            transcription = " "
            #raise ValueError("No transcription found in data.")
        return transcription
    
    def _prepare_sample_strategy(self):
        if self.config.get("sample_strategy") is None:
            return
        elif self.config.get("sample_strategy") == "sequential":
            self.audio_filepath2idx = {item["audio_filepath"]: i for i, item in enumerate(self.input_data)}
        else:
            key = self.config.get("sample_strategy").split("@")[1]
            group_by_data = defaultdict(list)
            for item in self.input_data:
                assert key in item, f"Key {key} not found in data"
                group_by_data[item[key]].append(item)
        
            self.group_by_data = group_by_data
    
    def sample(self, n, audio_filepath):
        if self.config.get("sample_strategy") is None or self.config.get("sample_strategy") == "random":
            return random.sample(self.input_data, n)
        
        elif self.config.get("sample_strategy") == "sequential":
            idx = self.audio_filepath2idx[audio_filepath]
            samples = self.input_data[idx+1:idx+n+1]
            if len(samples) < n:
                samples = samples + random.sample(self.input_data, n-len(samples))
            return samples
        
        elif self.config.get("sample_strategy") == "mixed":
            if random.random() < 0.5:
                return random.sample(self.input_data, n)
            elif random.random() < 0.90:
                idx = self.audio_filepath2idx[audio_filepath]
                samples = self.input_data[idx+1:idx+n+1]
                if len(samples) < n:
                    samples = samples + random.sample(self.input_data, n-len(samples))
                return samples
            else:
                idx = self.audio_filepath2idx[audio_filepath]
                samples = [self.input_data[idx]] * n
                return samples
            
        elif self.config.get("sample_strategy").startswith("group_by"):
            # key: group_by@speaker_id
            return random.sample(self.group_by_data[random.choice(list(self.group_by_data.keys()))], n)
        else:
            raise ValueError(f"Invalid sample strategy: {self.config.get('sample_strategy')}")


class MultiTurnMessageConstructor(MessagesConstructor):
    def __init__(self, config, tokenizer: AutoTokenizer, input_data: List[dict]):
        super().__init__(config, tokenizer, input_data)

    def construct_messages(self, datas: List[dict], messages, audio_locator: str = "<|AUDIO|>",):
        if self.config.get("system_prompts"):
            system_prompt = random.choice(self.config.get("system_prompts"))
        else:
            raise ValueError("No system prompt found in config")

        if datas[0].get("instruction") is not None:
            instruction = datas[0]["instruction"]
        elif datas[0].get("input") is not None:
            instruction = datas[0]["input"]
        elif self.config.get("instructions") is not None:
            instruction = random.choice(self.config["instructions"])
        else:
            raise ValueError("No instruction or input found in data")
        
        if messages is not None:
            template = deepcopy(messages)
        elif datas[0].get("messages") is not None:
            template = deepcopy(datas[0]["messages"])
        elif self.config.get("templates") is not None:
            template = deepcopy(random.choice(self.config.get("templates")))
        else:
            raise ValueError("No template messages found in data")

        seed_transcripts = [deepcopy(item["seed_transcript"]) for item in datas]

        new_messages = [] # replace <|AUDIO|> with seed_transcripts
        training_messages = [] # <|AUDIO|> messages for training
        audio_count = 0
        for message in template:
            message["content"] = message["content"].replace("{system_prompt}", system_prompt)
            message["content"] = message["content"].replace("{instruction}", instruction)

            training_messages.append(deepcopy(message))

            for _ in range(message["content"].count(audio_locator)):
                seed_transcript = seed_transcripts.pop(0)
                message["content"] = message["content"].replace(audio_locator, seed_transcript, 1)
                audio_count += 1

            new_messages.append(message)

        # assert audio_count == len(datas), f"audio_count: {audio_count}, len(datas): {len(datas)}"


        prompt = self.tokenizer.apply_chat_template(
                new_messages, tokenize=False, add_generation_prompt=True
            )
        return prompt, training_messages

