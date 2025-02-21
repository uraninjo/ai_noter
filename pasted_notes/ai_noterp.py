import sys
import os
current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(current_dir, "../"))

from utils import model_to_answer
from colorama import Fore
import json

sys.stdin.reconfigure(encoding='utf-8')  
sys.stdout.reconfigure(encoding='utf-8')  

username = os.environ.get("USER")
cache_dir = f"/home/{username}/.ai_noter_cache/paste_notes"
os.makedirs(cache_dir, exist_ok=True)
OUTPUT_FILE = f"{cache_dir}/notes.json"

if __name__ == "__main__":
    full_text = sys.stdin.read().strip()
    language = "tr"  
    USE_OLLAMA = False  

    if full_text:
        extracted_notes = model_to_answer(full_text)
        print(Fore.CYAN + "Notes: \n", extracted_notes)

        # JSON formatında kaydet
        output_data = {
            "full_text": full_text,
            "extracted_notes": extracted_notes,
            "language": language,
            "use_ollama": USE_OLLAMA
        }

        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)

        print(Fore.GREEN + f"📌 Notlar {OUTPUT_FILE} dosyasına kaydedildi.")
