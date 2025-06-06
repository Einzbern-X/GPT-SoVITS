import argparse
import os
import librosa
import numpy as np
import soundfile as sf
from time import time as ttime
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from module.models import SynthesizerTrn
from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from text import cleaned_text_to_sequence
from text.cleaner import clean_text
from my_utils import load_audio
import config as global_config

# Global configuration
g_config = global_config.Config()

# Setup argument parser
def setup_argparser():
    parser = argparse.ArgumentParser(description="Batch TTS Generation API")

    # Arguments for model paths and configuration
    parser.add_argument("-s", "--sovits_path", type=str, required=True, help="Path to SoVITS model")
    parser.add_argument("-g", "--gpt_path", type=str, required=True, help="Path to GPT model")
    parser.add_argument("-dr", "--default_refer_path", type=str, required=True, help="Path to the reference audio")
    parser.add_argument("-dt", "--default_refer_text", type=str, required=True, help="Reference audio text")
    parser.add_argument("-dl", "--default_refer_language", type=str, required=True, help="Language for reference audio")

    # Arguments for output and device configurations
    parser.add_argument("-d", "--device", type=str, default="cuda", help="Device to use (cuda/cpu)")
    parser.add_argument("-o", "--output_folder", type=str, required=True, help="Folder to save the output audio files")
    parser.add_argument("-t", "--text_file", type=str, required=True, help="Text file containing texts for TTS")

    return parser

# Main TTS generation function
def generate_audio(sovits_path, gpt_path, ref_wav_path, ref_text, ref_language, text_file, output_folder, device="cuda"):
    # Load models
    cnhubert_base_path = g_config.cnhubert_path
    bert_path = g_config.bert_path
    tokenizer = AutoTokenizer.from_pretrained(bert_path)
    bert_model = AutoModelForMaskedLM.from_pretrained(bert_path)

    # Load SoVITS model
    dict_s2 = torch.load(sovits_path, map_location="cpu", weights_only=False)
    hps = dict_s2["config"]
    vq_model = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model)
    vq_model.eval()
    vq_model.to(device)

    # Load GPT model
    dict_s1 = torch.load(gpt_path, map_location="cpu", weights_only=False)
    config = dict_s1["config"]
    t2s_model = Text2SemanticLightningModule(config, "ojbk", is_train=False)
    t2s_model.load_state_dict(dict_s1["weight"])
    t2s_model.to(device)
    t2s_model.eval()

    # Read input text file
    with open(text_file, "r", encoding="utf8") as f:
        text_lines = f.read().splitlines()

    # Process each line in the input text file
    for idx, text in enumerate(text_lines):
        print(f"Generating audio for line {idx+1}: {text}")
        audio = get_tts_wav(ref_wav_path, ref_text, ref_language, text, ref_language, device, hps, vq_model, t2s_model)

        # Save audio to output folder
        output_path = os.path.join(output_folder, f"{idx+1:04d}-{text[:10]}.wav")
        sf.write(output_path, audio, hps.data.sampling_rate)
        print(f"Saved audio to {output_path}")

# Function to generate TTS from text
def get_tts_wav(ref_wav_path, prompt_text, prompt_language, text, text_language, device, hps, vq_model, t2s_model):
    zero_wav = np.zeros(int(hps.data.sampling_rate * 0.3), dtype=np.float16)
    with torch.no_grad():
        wav16k, sr = librosa.load(ref_wav_path, sr=16000)
        wav16k = torch.from_numpy(wav16k).to(device)
        zero_wav_torch = torch.from_numpy(zero_wav).to(device)
        wav16k = torch.cat([wav16k, zero_wav_torch])

        # Generate semantic features from reference audio
        ssl_content = ssl_model.model(wav16k.unsqueeze(0))["last_hidden_state"].transpose(1, 2)
        codes = vq_model.extract_latent(ssl_content)
        prompt_semantic = codes[0, 0]

    phones, word2ph, norm_text = clean_text(text, text_language)
    phones = cleaned_text_to_sequence(phones)

    # Convert text to semantic features
    bert = get_bert_feature(norm_text, word2ph).to(device) if prompt_language == "zh" else torch.zeros((1024, len(phones)), dtype=torch.float32).to(device)

    all_phoneme_ids = torch.LongTensor(phones).to(device).unsqueeze(0)
    bert = bert.unsqueeze(0)
    all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(device)

    prompt = prompt_semantic.unsqueeze(0).to(device)
    with torch.no_grad():
        pred_semantic, idx = t2s_model.model.infer_panel(
            all_phoneme_ids, all_phoneme_len, prompt, bert,
            top_k=config['inference']['top_k'], early_stop_num=hz * max_sec)

    pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)
    refer = get_spepc(hps, ref_wav_path).to(device)
    audio = vq_model.decode(pred_semantic, torch.LongTensor(phones).to(device).unsqueeze(0), refer).detach().cpu().numpy()[0, 0]
    return (np.concatenate([audio, zero_wav]) * 32768).astype(np.int16)

# Function to parse arguments and execute the TTS process
def main():
    parser = setup_argparser()
    args = parser.parse_args()

    # Call generate audio function with the given parameters
    generate_audio(
        args.sovits_path, args.gpt_path,
        args.default_refer_path, args.default_refer_text, args.default_refer_language,
        args.text_file, args.output_folder, device=args.device
    )

if __name__ == "__main__":
    main()
