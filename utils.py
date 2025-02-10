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
        print("Alias already exists. No changes made.")
    else:
        with open(shell_rc, "a") as file:
            file.write(f"\n{alias_command}\n")
        print("Alias added successfully. Restart your shell or run 'source ~/.bashrc' (or ~/.zshrc) to apply changes.")

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
    
    updated_notes = initial_notes
    while True:
        print("\nChatbot: Güncellenmiş notlar:\n" if language == "tr" else "\nChatbot: Updated notes:\n", updated_notes)

        user_input = input("\nDüzenlemek için isteğinizi girin: " if language == "tr" else "\nEnter your request to edit: ")
        
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
        
        print("\nChatbot: Güncellenmiş notlar:\n" if language == "tr" else "\nChatbot: Updated notes:\n", updated_notes)
        
        initial_notes = updated_notes  # Notları güncelleyerek döngüye devam et
    return updated_notes

def get_prompt(language="tr"):
    if language == "tr":
        prompt = """
        Sana bir metin vereceğim. Bu metinden alınabilecek önemli notları çıkar. 
        Aşağıdaki adımları takip et:
        1. Önemli bilgileri ve maddeleri belirle.
        2. Konunun özetini çıkar.
        3. Eğer sayısal veriler veya istatistikler varsa, bunları listele.
        4. Eğer metinde eylem adımları varsa, bunları belirgin hale getir.
        
        Çıktıyı şu formatta ver:
        - **Özet:** (Metnin kısa özeti)
        - **Önemli Noktalar:** (Maddeler halinde bilgiler)
        - **İlgili Sayısal Veriler:** (Varsa istatistikler, sayılar)
        - **Eylem Adımları:** (Varsa talimatlar veya yapılması gerekenler)
        """
    else:
        prompt = """
        I will give you a text. Extract all key points that can be noted. 
        Follow these steps:
        1. Identify important pieces of information and key points.
        2. Summarize the main topic.
        3. If there are numerical data or statistics, list them.
        4. If there are action steps, highlight them.

        Provide the output in the following format:
        - **Summary:** (A brief summary of the text)
        - **Key Points:** (Bullet points of important details)
        - **Relevant Numerical Data:** (Any numbers or statistics)
        - **Actionable Steps:** (Instructions or steps if applicable)
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
    print("Notes: \n", answer)
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

    print("Notes: \n", answer)
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