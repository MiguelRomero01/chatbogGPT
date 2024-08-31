from flask import Flask, render_template, request, jsonify
from googletrans import Translator
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Cargar el modelo y el tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# Iniciar la app Flask
app = Flask(__name__)

# Inicializar el traductor de Google
translator = Translator()

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["POST"])
def chat():
    # Obtener el mensaje del usuario
    msg = request.form["msg"]
    #obtiene la respuesta del bot en español
    if translator.detect(msg).lang == "es":
        translated_response = translate(msg)
    #obtiene la respuesta del bot en ingles
    elif translator.detect(msg).lang == "en":
        translated_response = get_Chat_response(msg)
    
    return translated_response

"""
    funcion: si la entrada del usuario es en español, el bot responderá en español
    argumentos: msg de la chat()
"""
def translate(textToTranslate):
    detected_language = translator.detect(textToTranslate).lang
    
    try:
        # Traducir la entrada del usuario al inglés
        translated_input = translator.translate(textToTranslate, src='es', dest='en').text
        
        # Obtener la respuesta del bot en inglés
        response = get_Chat_response(translated_input)
        
        # Traducir la respuesta del bot al español
        translated_response = translator.translate(response, src='en', dest='es').text

    except Exception as e:
            print("Error al traducir: ",e)

    return translated_response
    

def get_Chat_response(text):
    # Codificar la entrada del usuario
    new_user_input_ids = tokenizer.encode(text + tokenizer.eos_token, return_tensors='pt')

    # Generar una respuesta
    chat_history_ids = model.generate(new_user_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # Decodificar y retornar la respuesta del bot
    return tokenizer.decode(chat_history_ids[:, new_user_input_ids.shape[-1]:][0], skip_special_tokens=True)

if __name__ == '__main__':
    app.run()
