import sys
import os
current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(current_dir, "../"))

import json
from utils import chatbot_interface
from colorama import Fore

username = os.environ.get("USER")
cache_dir = f"/home/{username}/.ai_noter_cache/paste_notes"
INPUT_FILE = f"{cache_dir}/notes.json"

if __name__ == "__main__":
    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(Fore.RED + f"Hata: {INPUT_FILE} bulunamadı. Önce 'pbpaste | python ai_noterp.py' çalıştırın.")
        sys.exit(1)

    full_text = data["full_text"]
    extracted_notes = data["extracted_notes"]
    language = data["language"]
    USE_OLLAMA = data["use_ollama"]

    # Chatbot ile düzenleme sürecine gir
    updated_notes = chatbot_interface(extracted_notes, full_text, language, use_ollama=USE_OLLAMA)

    # Güncellenmiş notları JSON dosyasına geri kaydet
    data["extracted_notes"] = updated_notes
    with open(INPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(Fore.GREEN + f"\n✅ Güncellenmiş notlar {INPUT_FILE} dosyasına kaydedildi.")
