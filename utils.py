import warnings as w
w.simplefilter("ignore")  # Import kaynaklı uyarıları (ör. google.generativeai FutureWarning) sustur

import yt_dlp
import os
from faster_whisper import WhisperModel
import google.generativeai as genai
import time
import ollama
import subprocess
import json
import re
import requests
from colorama import Fore, Style, init
import sys
import locale

sys.stdin.reconfigure(encoding='utf-8')  # input() için UTF-8 kodlamasını zorla
sys.stdout.reconfigure(encoding='utf-8')  # print() için UTF-8 kodlamasını zorla

# print("Terminal encoding:", locale.getpreferredencoding())  # Terminalin karakter setini gör

init(autoreset=True)

def get_API_KEY_env(key_name="GOOGLE_API_KEY"):
    if os.path.exists(".env"):
        with open(".env", "r") as f:
            for line in f.readlines():
                if key_name in line:
                    return line.split("=")[1].strip()

def setup_alias():
    project_dir = os.path.dirname(os.path.realpath(__file__))
    script_path = os.path.join(project_dir, "ai_noter.py")
    # Proje klasöründeki .venv ortamının python yorumlayıcısını kullan
    venv_python = os.path.join(project_dir, ".venv", "bin", "python")
    python_executable = venv_python if os.path.exists(venv_python) else (sys.executable or "python3")
    alias_command = f"alias ai_noter='{python_executable} {script_path}'"

    # Kullanıcının kabuğunu belirle ve uygun dosyayı seç
    shell_rc = os.path.expanduser("~/.bashrc") if os.path.exists(os.path.expanduser("~/.bashrc")) else os.path.expanduser("~/.zshrc")

    # Dosya içinde alias olup olmadığını kontrol et
    with open(shell_rc, "r") as file:
        lines = file.readlines()

    if any(alias_command.strip() == line.strip() for line in lines):
        print(Fore.YELLOW + "Alias already exists. No changes made.")
    else:
        # Farklı bir yorumlayıcıyla eklenmiş eski ai_noter alias satırlarını temizle
        new_lines = [line for line in lines if not line.strip().startswith("alias ai_noter=")]
        if new_lines and not new_lines[-1].endswith("\n"):
            new_lines[-1] += "\n"
        new_lines.append(f"{alias_command}\n")
        with open(shell_rc, "w") as file:
            file.writelines(new_lines)
        print(Fore.GREEN, "Alias added/updated successfully. Restart your shell or run 'source ~/.bashrc' (or ~/.zshrc) to apply changes.")

def check_ollama_models():
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError("Failed to execute the Ollama command. Ensure Ollama is installed and running.")
        output = result.stdout.strip()
        if not output:
            raise RuntimeError("Ollama command returned an empty response. Ensure Ollama is running.")
        # Çıktıyı satır bazında ayır
        lines = output.split("\n")
        # İlk satır başlık olduğu için onu atla
        model_lines = lines[1:]
        if not model_lines:
            raise RuntimeError("No models are installed in Ollama. Please install a model first.")
        model_names = [line.split("\t")[0] for line in model_lines]  # İlk sütun model adı
        return model_names

    except Exception as e:
        raise RuntimeError(f"Error while checking Ollama models: {e}")

def get_model(model_name='gemini-1.5-flash', prompt=None, language="tr", GOOGLE_API_KEY=None):
    if prompt is None:
        prompt = get_prompt(language=language)
    genai.configure(api_key=GOOGLE_API_KEY)
    return genai.GenerativeModel(model_name, system_instruction=prompt)

def get_chatbot_model(model_name='gemini-1.5-flash', prompt=None, language="tr", GOOGLE_API_KEY=None):
    if prompt is None:
        prompt = get_chatbot_prompt(language)
    genai.configure(api_key=GOOGLE_API_KEY)
    return genai.GenerativeModel(model_name, system_instruction=prompt)

def get_ollama_response(prompt, model="deepseek-r1:14b"):
    response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"] if "message" in response else "Hata oluştu."

def get_openrouter_response(prompt, system_prompt=None, model="tencent/hy3:free", api_key=None, reasoning=True):
    """OpenRouter chat/completions API'sine istek atar ve cevabı döndürür."""
    if api_key is None:
        api_key = os.getenv("OPENROUTER_API_KEY") or get_API_KEY_env("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is not set. Please set it in the environment variables or .env file.")

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    answer = None
    while answer is None:
        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                data=json.dumps({
                    "model": model,
                    "messages": messages,
                    "reasoning": {"enabled": reasoning},
                }),
            )
            response.raise_for_status()
            data = response.json()
            answer = data["choices"][0]["message"]["content"]
        except Exception as e:
            print("OpenRouter Modeli Hata", e)
            print("Tekrar denenmeden önce biraz bekleniyor...")
            time.sleep(10)
    return answer

def get_chatbot_prompt(language="tr"):
    if language == "tr":
        prompt = """
        Bir videonun metnini alıp, bununla bir not oluşturuldu. Bu notu ve metni sana vereceğim. 
        Bu notla ilgili sorulacak soruları yanıtlayacaksın
        """
    else:
        prompt = """
        A note was created by taking the text of a video. I will give you this note and text.
        You will answer questions about this note
        """
    return prompt

def chatbot_interface(initial_notes, full_text, language="tr", provider="openrouter", model_name=None, api_key=None):
    """
    Kullanıcı ile iteratif olarak notları geliştiren bir chatbot arayüzü.
    """
    conversation = f"Video'dan çıkarılan metin: {full_text}\nMevcut notlar:\n{initial_notes}\n" if language == "tr" else f"Extracted text from the video: {full_text}\nCurrent notes:\n{initial_notes}\n"
    
    if language == "tr":
        print("\nNotları düzenlemek için chatbot ile sohbet edebilirsiniz. Çıkmak için 'exit' yazın.\n")
    else:
        print("\nYou can chat with the chatbot to edit the notes. Type 'exit' to quit.\n")
    
    updated_notes = None
    while True:
        if not updated_notes is None:
            print(Fore.MAGENTA + ("\nChatbot: Güncellenmiş notlar:\n" if language == "tr" else "\nChatbot: Updated notes:\n"), updated_notes)
        print(Fore.MAGENTA + "[Chatbot]: ", end="")
        print(Fore.MAGENTA + ("Düzenlemek için isteğinizi girin: " if language == "tr" else "Enter your request to edit: "))
        print(Fore.CYAN + "[Kullanıcı]: ", end="")
        user_input = input().strip()
        
        if user_input.lower() == "exit":
            print("Chatbot oturumu kapatıldı." if language == "tr" else "Chatbot session ended.")
            break
        
        conversation += f"\nKullanıcı isteği: {user_input}\n" if language == "tr" else f"\nUser request: {user_input}\n"
        
        if provider == "ollama":
            updated_notes = get_ollama_response(conversation, model=model_name or "deepseek-r1:14b")
            updated_notes = remove_think_sections(updated_notes)
        elif provider == "gemini":
            model = get_chatbot_model(language=language)
            response = model.generate_content(conversation)
            updated_notes = response_to_answer(response)
        else:
            system_prompt = get_chatbot_prompt(language)
            updated_notes = get_openrouter_response(conversation, system_prompt=system_prompt, model=model_name or "tencent/hy3:free", api_key=api_key)
            updated_notes = remove_think_sections(updated_notes)

        initial_notes = updated_notes  # Notları güncelleyerek döngüye devam et
    return updated_notes

def get_prompt(language="tr"):
    if language == "tr":
        prompt = """
        Sana bir metin vereceğim. Bu metinden alınabilecek en önemli, kritik ve detaylı bilgileri çıkar.
        Özellikle kendini geliştirmek isteyen bir insanın yararlanabileceği pratik öneri, strateji ve metodolojileri vurgula.
        Analiz yaparken aşağıdaki adımları takip et:

        1. **Kritik Bilgileri Belirle:**
           - Metindeki temel kavramları, önemli terimleri ve ana fikirleri tespit et.
           - İddiaları, savunulan görüşleri ve karşıt argümanları belirle.
           - Teknik veya akademik metinlerde kilit prensipleri ve süreçleri açıkla.

        2. **Derinlemesine Açıklama Yap:**
           - "Neden?", "Nasıl?", "Kim?", "Ne zaman?" ve "Ne şekilde?" sorularını sorarak bilgileri detaylandır.
           - Kavramların ve olayların arkasındaki nedenleri ve süreçleri açıklığa kavuştur.
           - "Bu bilgi neden kritiktir?" sorusuna yanıt vererek bilginin önemini vurgula.

        3. **Sayısal Verileri Listele:**
           - Metindeki istatistikleri, oranları, ölçüleri ve diğer sayısal bilgileri belirle.
           - Bu sayıların ne anlama geldiğini ve bağlamını açıkla.

        4. **Eyleme Geçirilebilir Adımları Vurgula:**
           - Herhangi bir süreç, prosedür, talimat veya uygulanabilir öneri varsa bunları listele.
           - Adımları açık, net ve sıralı olarak belirt.

        5. **Farklı Bakış Açılarını Dahil Et:**
           - Tartışma içeren metinlerde, farklı görüşleri listele ve güçlü/zayıf yönlerini analiz et.
           - Olası önyargıları veya eksik yönleri tespit et.

        6. **Duygusal ve Psikolojik Analiz:**
           - Metindeki duygusal ifadeleri, motivasyonları ve psikolojik durumları belirle.
           - Bu duyguların ve motivasyonların bilginin aktarımındaki etkisini yorumla.

        7. **Önemli Alıntıları (Quote) Belirle:**
           - Metin içinde yer alan, anlam taşıyan önemli alıntıları tespit et.
           - Alıntıların bağlamını ve neden kritik olduklarını kısaca açıkla.

        8. **Ayrı Konuların İncelemesini Sağla:**
           - Metinde birbirinden bağımsız veya farklı konular/bölümler varsa, bunları ayrı başlıklar altında değerlendir.
           - Her bölümün kendi kritik bilgilerini, eylem adımlarını ve detaylarını eksiksiz sun.

        9. **Özetleme Yaparken Detayları Koruyarak Özetle:**
           - Genel özet kısmını oluştururken ana noktaların kaybolmamasına dikkat et.
           - Özet, bilgiyi aşırı kısaltmadan, tüm kritik ve detaylı noktaları kapsayacak şekilde olmalı.

        10. **Kişisel Gelişim İçin Uygulanabilir Bilgiler:**
            - Metindeki, kendini geliştirmek isteyen bir bireyin yararlanabileceği pratik öneriler, stratejiler ve metodolojileri vurgula.
            - Her bilgiyi, kişisel gelişime nasıl katkıda bulunabileceğini belirterek açıkla.

        Çıktıyı aşağıdaki formatta ver:

        - **Özet:** (Metnin en önemli noktalarını içeren kısa ama detaylı bir özet)
        - **Önemli Noktalar:** (Maddeler halinde detaylı analiz)
        - **İlgili Sayısal Veriler:** (Varsa istatistikler ve sayılar)
        - **Eylem Adımları:** (Varsa uygulanabilir adımlar ve süreçler)
        - **Farklı Bakış Açıları:** (Varsa karşıt görüşler ve analizleri)
        - **Duygusal Analiz:** (Varsa metindeki duygusal ifadeler ve etkileri)
        - **Önemli Alıntılar:** (Metinden kritik anlam taşıyan alıntılar ve kısa açıklamaları)
        """
    else:
        prompt = """
        I will give you a text. Extract the most important, critical, and detailed information.
        Emphasize actionable insights, strategies, and methodologies that can benefit someone looking to improve themselves.
        Follow these steps for in-depth analysis:

        1. **Identify Critical Information:**
           - Detect key concepts, important terms, and core ideas in the text.
           - Identify claims, arguments, and counterarguments.
           - For technical or academic texts, explain the key principles and processes.

        2. **Provide Detailed Explanations:**
           - Elaborate by answering "Why?", "How?", "Who?", "When?", and "In what way?".
           - Explain the underlying reasons and mechanisms behind the concepts and events.
           - Emphasize the significance by answering "Why is this information critical?"

        3. **List Numerical Data:**
           - Identify any statistics, ratios, measurements, and numerical details.
           - Explain the meaning and context of these numbers.

        4. **Highlight Actionable Steps:**
           - If the text includes processes, procedures, instructions, or practical recommendations, list them.
           - Provide clear, ordered steps.

        5. **Include Different Perspectives:**
           - For texts that contain debates, list opposing viewpoints and analyze their strengths and weaknesses.
           - Identify any potential biases or missing elements.

        6. **Emotional and Psychological Analysis:**
           - Identify emotional expressions, motivations, or psychological states present in the text.
           - Comment on the impact and significance of these emotions or motivations on the overall message.

        7. **Identify Important Quotes:**
           - Extract key quotes from the text that carry significant meaning.
           - Briefly explain the context and why these quotes are critical.

        8. **Separate Analysis of Distinct Topics:**
           - If the text covers multiple independent topics or sections, analyze each separately.
           - Ensure each topic’s critical information, actionable steps, and details are clearly presented.

        9. **Avoid Over-Summarizing:**
           - In the summary section, ensure that the main points are not overly condensed to the point of losing critical details.
           - The summary should capture all essential and nuanced points.

        10. **Actionable Insights for Self-Improvement:**
            - Highlight practical suggestions, strategies, and methodologies that would benefit someone aiming for self-improvement.
            - Explain how each piece of information can contribute to personal development.

        Provide the output in the following format:

        - **Summary:** (A brief summary highlighting the key points without losing important details)
        - **Key Points:** (Bullet points with detailed analysis)
        - **Relevant Numerical Data:** (Any statistics or numbers if available)
        - **Actionable Steps:** (Any instructions or processes)
        - **Different Perspectives:** (If applicable, opposing views and analysis)
        - **Emotional Analysis:** (If applicable, any emotional expressions and their impact)
        - **Important Quotes:** (Key quotes from the text along with brief explanations)
        """
    return prompt


def response_to_answer(response):
    try:
        return response.text
    except Exception as e:
        print("Modelden cevap alınamadı. Hata:", e)
        return None

def model_to_answer(full_text, model_name='gemini-1.5-flash', prompt=None, language="tr"):
    model = get_model(model_name=model_name, prompt=prompt, language=language)
    # print("Model girdisi: ", full_text)
    response = None
    while response is None:
        try:
            response = model.generate_content(full_text)
        except Exception as e:
            print("Gemini Modeli Hata", e)
            print("Tekrar denenmeden önce biraz bekleniyor...")
            time.sleep(10)
    answer = response_to_answer(response)
    return answer

def remove_think_sections(text):
    """Metindeki <think>...</think> bloklarını temizler."""
    cleaned_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned_text.strip()

def model_to_answer_ollama(full_text, model_name='mistral', prompt=None, language="tr"):
    if prompt is None:
        prompt = get_prompt(language)
    
    response = None
    while response is None:
        try:
            response = ollama.generate(model=model_name, prompt=f"{prompt}\n\n{full_text}")
        except Exception as e:
            print("Ollama Modeli Hata", e)
            print("Tekrar denenmeden önce biraz bekleniyor...")
            time.sleep(10)
    
    answer = response["response"]
    answer = remove_think_sections(answer)

    return answer

def model_to_answer_openrouter(full_text, model_name='tencent/hy3:free', prompt=None, language="tr", api_key=None):
    if prompt is None:
        prompt = get_prompt(language)
    answer = get_openrouter_response(full_text, system_prompt=prompt, model=model_name, api_key=api_key)
    answer = remove_think_sections(answer)
    return answer

def model_to_answer_choose(full_text, model_name='gemini-1.5-flash', prompt=None, language="tr", provider="openrouter", api_key=None):
    if provider == "ollama":
        return model_to_answer_ollama(full_text, model_name=model_name, prompt=prompt, language=language)
    elif provider == "gemini":
        return model_to_answer(full_text, model_name=model_name, prompt=prompt, language=language)
    else:
        return model_to_answer_openrouter(full_text, model_name=model_name, prompt=prompt, language=language, api_key=api_key)

def download_audio_from_youtube(url, video_cache_path):
    os.makedirs(video_cache_path, exist_ok=True)
    
    # Video bilgilerini çek
    try:
        with yt_dlp.YoutubeDL({'quiet': True, 'no_warnings': True}) as ydl:
            info = ydl.extract_info(url, download=False)
            video_id = info.get("id", url.split("v=")[-1])
            title = info.get("title", "unknown_title")
    except Exception as e:
        print("Video bilgileri alınamadı:", e)
        return None, None

    output_path = os.path.join(video_cache_path, "audio_files", f"{video_id}")

    if os.path.exists(output_path + ".mp3"):
        return output_path + ".mp3", title
    elif os.path.exists(output_path + ".m4a"):
        return output_path + ".m4a", title

    # Download audio separately and extract it as mp3
    try:
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': output_path,
            'quiet': True,
            'no_warnings': True
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return output_path + ".mp3", title

    except Exception as e:
        print("Hata:", e)
        print("mp3 formatında indirme başarısız. m4a formatında indiriliyor.")
        
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'aac',
                'preferredquality': '192',
            }],
            'outtmpl': output_path,
            'quiet': False,
            'no_warnings': True
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return output_path + ".m4a", title

def run_whisper(input_file= "audio.mp3", word_timestamps=False, model_name="large_v3"):
    model = WhisperModel(model_name, device="cuda", compute_type="float16")
    segments, info = model.transcribe(input_file, vad_filter=True, vad_parameters=dict(min_silence_duration_ms=100), word_timestamps=word_timestamps)
    # segments, info = model.detect_language_multi_segment(input_file)
    # print("Detected language '{}' with probability {:.2f}".format(info.language, info.language_probability))
    return segments, info.language

def print_segments(segments, log=False):
    word_by_word = []
    senteces = []
    segments = list(segments)

    for segment in segments:
        for word in segment.words:
            if log:
                print("[%.2fs -> %.2fs] %s" % (word.start, word.end, word.word))
            word_by_word.append([word.start, word.end, word.word])

        if log:
            print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
        senteces.append([segment.start, segment.end, segment.text])
    return word_by_word, senteces