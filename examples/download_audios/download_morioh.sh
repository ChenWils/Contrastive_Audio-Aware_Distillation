# For loop path_in_repo

export HF_TOKEN=your_hf_token_here

for path_in_repo in IEMOCAP Anispeech EmoV_DB Emotion_Speech_Dataset GLOBE_V2 CAFE NTUML2021 Clotho AudioSet-20K EMOVO AudioCaps cszs_es_en Nsynth voxceleb1 L2Arctic Audiocaps2 GtzanGenre Voxlingual_Top10 cszs_zh_en EMNS ASVspoof5 CodecFake MSP_IMPROV TMHINT-QI ESC50 Libricount AccentDB_extended cszs_fr_en MELD CREMA-D VCTK-Corpus FMA_medium .gitattributes VCTK_augmented2 Mridangam VocalSound VCTK_augmented common_voice_zh dynamic-superb-noise-reverb expresso Dailytalk ASVspoofing2019 LibriSpeech-c ASVspoof2015 PromptTTS KeSpeech meta_fair_asr common_voice_en speech_accent_archive FSD50K Speech_Command
# for path_in_repo in expresso voxceleb1 MELD
# for path_in_repo in CAFE
do
    echo ===== $path_in_repo =====
    echo Downloading...
    python /home/jovyan/shared/kehanluu/workspace/DeSTA3-dev/examples/download_audios/download_from_hf.py --repo_id Morioh/livingroom --revision desta --path_in_repo $path_in_repo --data_root /home/jovyan/workspace/data/audios --stage download
    echo Extracting...
    python /home/jovyan/shared/kehanluu/workspace/DeSTA3-dev/examples/download_audios/download_from_hf.py --repo_id Morioh/livingroom --revision desta --path_in_repo $path_in_repo --data_root /home/jovyan/workspace/data/audios --stage extract
    echo Done
done
