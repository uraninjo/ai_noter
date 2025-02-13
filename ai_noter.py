import os
import time
import pickle as pkl
import argparse
from utils import download_audio_from_youtube, run_whisper, print_segments, model_to_answer_choose, chatbot_interface, get_API_KEY_env, check_ollama_models, setup_alias
from colorama import Fore, Style, init
import sys
import locale

sys.stdin.reconfigure(encoding='utf-8')  # input() için UTF-8 kodlamasını zorla
sys.stdout.reconfigure(encoding='utf-8')  # print() için UTF-8 kodlamasını zorla
init(autoreset=True)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # TensorFlow uyarılarını kapatır
os.environ["GRPC_VERBOSITY"] = "ERROR"  # gRPC hata mesajlarını gizler
os.environ["GRPC_TRACE"] = ""  # gRPC detaylı loglarını kapatır

os.environ["TOKENIZERS_PARALLELISM"] = "False"

def main():
    setup_alias()
    parser = argparse.ArgumentParser(description="A script that downloads audio from YouTube videos and converts it to text.")
    parser.add_argument("youtube_url", type=str, help="YouTube URL to be processed")
    parser.add_argument("--use_ollama", action="store_true", help="Enable Ollama usage (optional).")
    parser.add_argument("--ollama_model_name", type=str, default="deepseek-r1:14b", help="Ollama model to be used (default: deepseek-r1:14b)")
    parser.add_argument("--whisper_model_size", type=str, default="base", help="Whisper model to be used (default: base)")
    args = parser.parse_args()
    
    username = os.environ.get("USER")
    video_cache_path = f"/home/{username}/.ai_noter_cache"
    os.makedirs(video_cache_path, exist_ok=True)
    
    # yt_start_time = time.time()
    WHISPER_MODEL_NAME = args.whisper_model_size
    VIDEO_URL = args.youtube_url
    USE_OLLAMA = args.use_ollama
    
    if not USE_OLLAMA:
        GOOGLE_API_KEY = "" or os.getenv("GOOGLE_API_KEY") or get_API_KEY_env()

        if GOOGLE_API_KEY == "":
            raise "GOOGLE_API_KEY is not set. Please set it in the environment variables."
        else:
            LLM_MODEL_NAME = "gemini-1.5-flash"
    else:
        # Check models and ensure the specified model is installed
        installed_models = check_ollama_models()
        if args.ollama_model_name not in installed_models:
            raise ValueError(f"The specified model '{args.ollama_model_name}' is not installed in Ollama.")
        else:
            LLM_MODEL_NAME = args.ollama_model_name

    audio_path, title = download_audio_from_youtube(VIDEO_URL, video_cache_path)
    print(Fore.GREEN + f"Video name:\n {title}\n")
    
    video_name = os.path.basename(audio_path).split(".")[0]

    whisper_pkl_path = os.path.join(video_cache_path, f"{video_name}_segments.pkl")
    
    if os.path.exists(whisper_pkl_path):
        with open(whisper_pkl_path, "rb") as file:
            word_by_word_segments, segments, language = pkl.load(file)
    else:
        word_segments, language = run_whisper(input_file=audio_path, word_timestamps=True, model_name=WHISPER_MODEL_NAME)
        word_by_word_segments, segments = print_segments(word_segments)
        pkl.dump([word_by_word_segments, segments, language], open(whisper_pkl_path, "wb"))
    
    full_text = "".join([segment[2] for segment in segments])

    extracted_notes = model_to_answer_choose(full_text, model_name=LLM_MODEL_NAME, prompt=None, language=language, USE_OLLAMA=USE_OLLAMA)
    print(Fore.CYAN + "Notes: \n", extracted_notes)

    note_path = os.path.join(video_cache_path, f"{video_name}_notes.txt")

    print(Fore.MAGENTA + "[Chatbot]: Do you want AI to edit the notes (yes/no):")
    print(Fore.CYAN + "[Kullanıcı]: ", end="")
    user_feedback = input()
    if user_feedback.lower() == "y" or user_feedback.lower() == "yes":
        extracted_notes = chatbot_interface(extracted_notes, full_text, language, use_ollama=USE_OLLAMA)
    else:
        print(Fore.GREEN, "Notes saved.")

    with open(note_path, "w") as file:
        file.write(extracted_notes)
    
if __name__ == "__main__":
    main()
