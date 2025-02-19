import yt_dlp
import warnings as w
import os
from faster_whisper import WhisperModel
import google.generativeai as genai
import time
import ollama
import subprocess
import json
import re
from colorama import Fore, Style, init
import sys
import locale

sys.stdin.reconfigure(encoding='utf-8')  # input() için UTF-8 kodlamasını zorla
sys.stdout.reconfigure(encoding='utf-8')  # print() için UTF-8 kodlamasını zorla

# print("Terminal encoding:", locale.getpreferredencoding())  # Terminalin karakter setini gör

init(autoreset=True)
w.simplefilter("ignore")

def get_API_KEY_env():
    if os.path.exists(".env"):
        with open(".env", "r") as f:
            for line in f.readlines():
                if "GOOGLE_API_KEY" in line:
                    return line.split("=")[1].strip()

def setup_alias():
    script_path = os.path.dirname(os.path.realpath(__file__)) + "/ai_noter.py"
    alias_command = f"alias ai_noter='python {script_path}'"
    
    # Kullanıcının kabuğunu belirle ve uygun dosyayı seç
    shell_rc = os.path.expanduser("~/.bashrc") if os.path.exists(os.path.expanduser("~/.bashrc")) else os.path.expanduser("~/.zshrc")

    # Dosya içinde alias olup olmadığını kontrol et
    with open(shell_rc, "r") as file:
        lines = file.readlines()
    
    if any(alias_command.strip() in line.strip() for line in lines):  
        print(Fore.YELLOW + "Alias already exists. No changes made.")
    else:
        with open(shell_rc, "a") as file:
            file.write(f"\n{alias_command}\n")
        print(Fore.GREEN, "Alias added successfully. Restart your shell or run 'source ~/.bashrc' (or ~/.zshrc) to apply changes.")

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

def chatbot_interface(initial_notes, full_text, language="tr", use_ollama=True):
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
        
        if use_ollama:
            updated_notes = get_ollama_response(conversation)
            updated_notes = remove_think_sections(updated_notes)
        else:
            model = get_chatbot_model(language=language)
            response = model.generate_content(conversation)
            updated_notes = response_to_answer(response)

        initial_notes = updated_notes  # Notları güncelleyerek döngüye devam et
    return updated_notes

def get_prompt(language="tr"):
    if language == "tr":
        prompt = """
        Sana bir metin vereceğim. Bu metinden alınabilecek en önemli, kritik ve detaylı bilgileri çıkar. 
        Analiz yaparken şu adımları takip et:
        
        1. **Kritik Bilgileri Belirle:**  
            - Metindeki temel kavramları, önemli terimleri ve ana fikirleri tespit et.  
            - Metindeki iddiaları, savunulan görüşleri ve karşıt argümanları belirle.  
            - Eğer metin teknik veya akademikse, kilit prensipleri ve süreçleri açıkla.  
        
        2. **Derinlemesine Açıklama Yap:**  
            - "Neden?", "Nasıl?", "Kim?", "Ne zaman?" ve "Ne şekilde?" sorularını sorarak bilgileri detaylandır.  
            - Kavramların ve olayların arkasındaki nedenleri ve süreçleri açıkla.  
            - Bilginin önemini vurgula: "Bu bilgi neden kritiktir?" sorusuna yanıt ver.  
        
        3. **Sayısal Verileri Listele:**  
            - Metinde geçen istatistikleri, oranları, ölçüleri ve diğer sayısal bilgileri belirle.  
            - Sayısal verileri bağlam içinde açıkla: "Bu sayı ne anlama geliyor?" sorusunu yanıtla.  
        
        4. **Eyleme Geçirilebilir Adımları Vurgula:**  
            - Metin içinde herhangi bir süreç, prosedür, talimat veya uygulanabilir bir öneri varsa, bunları listele.  
            - Adımları açık, net ve sıralı olarak belirt.  
        
        5. **Farklı Bakış Açılarını Dahil Et:**  
            - Eğer metin bir tartışma içeriyorsa, farklı görüşleri listele ve güçlü/zayıf yönlerini açıkla.  
            - Metindeki olası önyargıları veya eksik yönleri tespit et.  

        Çıktıyı şu formatta ver:

        - **Özet:** (Metnin en önemli noktalarını içeren kısa bir özet)  
        - **Önemli Noktalar:** (Maddeler halinde detaylı analiz)  
        - **İlgili Sayısal Veriler:** (Varsa istatistikler ve sayılar)  
        - **Eylem Adımları:** (Varsa yapılması gerekenler)  
        - **Farklı Bakış Açıları:** (Varsa karşıt görüşler ve analizleri)  
        """
    else:
        prompt = """
        I will give you a text. Extract the most important, critical, and detailed information.  
        Follow these steps for in-depth analysis:
        
        1. **Identify Critical Information:**  
            - Detect key concepts, important terms, and core ideas.  
            - Identify claims, arguments, and counterarguments presented in the text.  
            - If the text is technical or academic, explain the key principles and processes.  
        
        2. **Provide Detailed Explanations:**  
            - Answer "Why?", "How?", "Who?", "When?", and "In what way?" questions.  
            - Explain the reasons and mechanisms behind concepts and events.  
            - Highlight the significance of the information: "Why is this important?"  

        3. **List Numerical Data:**  
            - Identify statistics, ratios, measurements, and other numerical data.  
            - Explain the context of numbers: "What does this number mean?"  

        4. **Highlight Actionable Steps:**  
            - If the text includes a process, procedure, instructions, or practical recommendations, list them.  
            - Provide steps in a clear and ordered manner.  

        5. **Include Different Perspectives:**  
            - If the text contains debates, list opposing views and analyze their strengths/weaknesses.  
            - Detect potential biases or missing elements in the discussion.  

        Provide the output in the following format:

        - **Summary:** (A brief summary highlighting key points)  
        - **Key Points:** (Bullet points with detailed analysis)  
        - **Relevant Numerical Data:** (Any numbers or statistics if available)  
        - **Actionable Steps:** (If there are any instructions or processes)  
        - **Different Perspectives:** (If applicable, opposing views and analysis)  
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

def model_to_answer_choose(full_text, model_name='gemini-1.5-flash', prompt=None, language="tr", USE_OLLAMA=False):
    if USE_OLLAMA:
        return model_to_answer_ollama(full_text, model_name=model_name, prompt=prompt, language=language)

    else:
        return model_to_answer(full_text, model_name=model_name, prompt=prompt, language=language)

def download_audio_from_youtube(url, video_cache_path):
    os.makedirs(video_cache_path, exist_ok=True)
    
    # Video bilgilerini çek
    try:
        with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
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
            'quiet': True
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
            'quiet': False
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