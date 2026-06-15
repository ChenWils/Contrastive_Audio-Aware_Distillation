import numpy as np

# Data mapping for DS scores
ds_scores = {
    "DS@DialogueActPairing_DailyTalk": 0.455,
    "DS@SpeakerVerification_LibriSpeech-TestClean": 0.49,
    "DS@SpeakerVerification_VCTK": 0.475,
    "DS@AccentClassification_AccentdbExtended": 0.725,
    "DS@BirdSoundDetection_Warblrb10k": 0.715,
    "DS@ChordClassification_AcousticGuitarAndPiano": 0.525,
    "DS@DialogueActClassification_DailyTalk": 0.5,
    "DS@DialogueEmotionClassification_DailyTalk": 0.825,
    "DS@EmotionRecognition_MultimodalEmotionlinesDataset": 0.495,
    "DS@EnhancementDetection_LibriTTS-TestClean_WHAM": 0.42,
    "DS@EnvironmentalSoundClassification_ESC50-Animals": 0.175,
    "DS@EnvironmentalSoundClassification_ESC50-ExteriorAndUrbanNoises": 0.195,
    "DS@EnvironmentalSoundClassification_ESC50-HumanAndNonSpeechSounds": 0.235,
    "DS@EnvironmentalSoundClassification_ESC50-InteriorAndDomesticSounds": 0.175,
    "DS@EnvironmentalSoundClassification_ESC50-NaturalSoundscapesAndWaterSounds": 0.155,
    "DS@HowFarAreYou_3DSpeaker": 0.28,
    "DS@IntentClassification_FluentSpeechCommands-Action": 0.68,
    "DS@IntentClassification_FluentSpeechCommands-Location": 0.66,
    "DS@IntentClassification_FluentSpeechCommands-Object": 0.825,
    "DS@LanguageIdentification_VoxForge": 0.875,
    "DS@MultiSpeakerDetection_LibriSpeech-TestClean": 0.555,
    "DS@MultiSpeakerDetection_VCTK": 0.555,
    "DS@NoiseDetection_LJSpeech_MUSAN-Gaussian": 0.835,
    "DS@NoiseDetection_LJSpeech_MUSAN-Music": 0.5,
    "DS@NoiseDetection_LJSpeech_MUSAN-Noise": 0.47,
    "DS@NoiseDetection_LJSpeech_MUSAN-Speech": 0.49,
    "DS@NoiseDetection_VCTK-MUSAN-Gaussian": 0.45,
    "DS@NoiseDetection_VCTK_MUSAN-Music": 0.365,
    "DS@NoiseDetection_VCTK_MUSAN-Noise": 0.44,
    "DS@NoiseDetection_VCTK_MUSAN-Speech": 0.285,
    "DS@NoiseSNRLevelPrediction_VCTK_MUSAN-Gaussian": 0.245,
    "DS@NoiseSNRLevelPrediction_VCTK_MUSAN-Music": 0.215,
    "DS@NoiseSNRLevelPrediction_VCTK_MUSAN-Noise": 0.21,
    "DS@NoiseSNRLevelPrediction_VCTK_MUSAN-Speech": 0.175,
    "DS@ReverberationDetection_LJSpeech_RirsNoises-LargeRoom": 0.545,
    "DS@ReverberationDetection_LJSpeech_RirsNoises-MediumRoom": 0.545,
    "DS@ReverberationDetection_LJSpeech_RirsNoises-SmallRoom": 0.6,
    "DS@ReverberationDetection_VCTK_RirsNoises-LargeRoom": 0.47,
    "DS@ReverberationDetection_VCTK_RirsNoises-MediumRoom": 0.485,
    "DS@ReverberationDetection_VCTK_RirsNoises-SmallRoom": 0.56,
    "DS@SarcasmDetection_Mustard": 0.51,
    "DS@SpeakerCounting_LibriTTS-TestClean": 0.165,
    "DS@SpeechCommandRecognition_GoogleSpeechCommandsV1": 0.66,
    "DS@SpeechDetection_LJSpeech": 0.535,
    "DS@SpeechDetection_LibriSpeech-TestClean": 0.48,
    "DS@SpeechDetection_LibriSpeech-TestOther": 0.53,
    "DS@SpeechTextMatching_LJSpeech": 0.755,
    "DS@SpeechTextMatching_LibriSpeech-TestClean": 0.65,
    "DS@SpeechTextMatching_LibriSpeech-TestOther": 0.685,
    "DS@SpokenTermDetection_LJSpeech": 0.955,
    "DS@SpokenTermDetection_LibriSpeech-TestClean": 0.835,
    "DS@SpokenTermDetection_LibriSpeech-TestOther": 0.775,
    "DS@SpoofDetection_ASVspoof2015": 0.855,
    "DS@SpoofDetection_ASVspoof2017": 0.835,
    "DS@StressDetection_MIRSD": 0.04,
    "DS@DialogueActPairing_DailyTalk-multiaudio": 0.415,
    "DS@SpeakerVerification_LibriSpeech-TestClean-multiaudio": 0.23,
    "DS@SpeakerVerification_VCTK-multiaudio": 0.415
}

# Categories
categories = {
    "CON": [
        "DS@SpeechCommandRecognition_GoogleSpeechCommandsV1",
        "DS@SpokenTermDetection_LJSpeech",
        "DS@SpokenTermDetection_LibriSpeech-TestClean",
        "DS@SpokenTermDetection_LibriSpeech-TestOther",
        "DS@SpeechTextMatching_LJSpeech",
        "DS@SpeechTextMatching_LibriSpeech-TestClean",
        "DS@SpeechTextMatching_LibriSpeech-TestOther",
        "DS@LanguageIdentification_VoxForge",
        "DS@SpeechDetection_LJSpeech",
        "DS@SpeechDetection_LibriSpeech-TestClean",
        "DS@SpeechDetection_LibriSpeech-TestOther",
    ],
    "SEM": [
        "DS@IntentClassification_FluentSpeechCommands-Action",
        "DS@IntentClassification_FluentSpeechCommands-Location",
        "DS@IntentClassification_FluentSpeechCommands-Object",
        "DS@DialogueActClassification_DailyTalk",
        "DS@SarcasmDetection_Mustard",
        "DS@DialogueActPairing_DailyTalk",
        "DS@DialogueActPairing_DailyTalk-multiaudio",
    ],
    "PAR": [
        "DS@DialogueEmotionClassification_DailyTalk",
        "DS@EmotionRecognition_MultimodalEmotionlinesDataset",
        "DS@AccentClassification_AccentdbExtended",
        "DS@StressDetection_MIRSD",
        "DS@HowFarAreYou_3DSpeaker",
        "DS@SpoofDetection_ASVspoof2015",
        "DS@SpoofDetection_ASVspoof2017",
    ],
    "DEG": [
        "DS@NoiseDetection_LJSpeech_MUSAN-Gaussian",
        "DS@NoiseDetection_LJSpeech_MUSAN-Music",
        "DS@NoiseDetection_LJSpeech_MUSAN-Noise",
        "DS@NoiseDetection_LJSpeech_MUSAN-Speech",
        "DS@NoiseDetection_VCTK-MUSAN-Gaussian",
        "DS@NoiseDetection_VCTK_MUSAN-Music",
        "DS@NoiseDetection_VCTK_MUSAN-Noise",
        "DS@NoiseDetection_VCTK_MUSAN-Speech",
        "DS@NoiseSNRLevelPrediction_VCTK_MUSAN-Gaussian",
        "DS@NoiseSNRLevelPrediction_VCTK_MUSAN-Music",
        "DS@NoiseSNRLevelPrediction_VCTK_MUSAN-Noise",
        "DS@NoiseSNRLevelPrediction_VCTK_MUSAN-Speech",
        "DS@ReverberationDetection_LJSpeech_RirsNoises-LargeRoom",
        "DS@ReverberationDetection_LJSpeech_RirsNoises-MediumRoom",
        "DS@ReverberationDetection_LJSpeech_RirsNoises-SmallRoom",
        "DS@ReverberationDetection_VCTK_RirsNoises-LargeRoom",
        "DS@ReverberationDetection_VCTK_RirsNoises-MediumRoom",
        "DS@ReverberationDetection_VCTK_RirsNoises-SmallRoom",
        "DS@EnhancementDetection_LibriTTS-TestClean_WHAM",
    ],
    "SPK": [
        "DS@SpeakerVerification_LibriSpeech-TestClean",
        "DS@SpeakerVerification_VCTK",
        "DS@SpeakerVerification_LibriSpeech-TestClean-multiaudio",
        "DS@SpeakerVerification_VCTK-multiaudio",
        "DS@MultiSpeakerDetection_LibriSpeech-TestClean",
        "DS@MultiSpeakerDetection_VCTK",
        "DS@SpeakerCounting_LibriTTS-TestClean",
    ],
    "AUDIO":[
        "DS@EnvironmentalSoundClassification_ESC50-Animals",
        "DS@EnvironmentalSoundClassification_ESC50-ExteriorAndUrbanNoises",
        "DS@EnvironmentalSoundClassification_ESC50-HumanAndNonSpeechSounds",
        "DS@EnvironmentalSoundClassification_ESC50-InteriorAndDomesticSounds",
        "DS@EnvironmentalSoundClassification_ESC50-NaturalSoundscapesAndWaterSounds",
        "DS@ChordClassification_AcousticGuitarAndPiano",
        "DS@BirdSoundDetection_Warblrb10k",
    ],
}

# Calculating the average score for each category
category_averages = {}
for category, items in categories.items():
    scores = [ds_scores[item] for item in items]
    category_averages[category] = np.mean(scores)

print("Category Averages:")
print(category_averages)

# Calculate overall average excluding AUDIO category (average of all individual tasks)
all_non_audio_scores = []
for category, items in categories.items():
    if category != "AUDIO":
        all_non_audio_scores.extend([ds_scores[item] for item in items])

overall_average_without_audio = np.mean(all_non_audio_scores)

print(f"\nOverall Average (without AUDIO, all {len(all_non_audio_scores)} tasks): {overall_average_without_audio:.4f}")
