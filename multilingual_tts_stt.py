# backend/multilingual_tts_stt.py
import whisper
import speech_recognition as sr
from gtts import gTTS
import os
from utils.translator import translate_text

model = whisper.load_model("base")

def transcribe_voice(file_path, input_lang):
    result = model.transcribe(file_path, language=input_lang)
    return result["text"]

def text_to_speech(text, lang):
    tts = gTTS(text=translate_text(text, 'en', lang), lang=lang)
    out_path = f"output_{lang}.mp3"
    tts.save(out_path)
    return out_path
