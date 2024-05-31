from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

# A simple in-memory store for translation history
translation_history = []

def pseudo_translate(text, lang):
    if lang == 'en_to_vi':
        # Placeholder translation logic: Prefix with "Translated to Vietnamese: "
        return f"Translated to Vietnamese: {text}"
    else:
        # Placeholder translation logic: Prefix with "Translated to English: "
        return f"Translated to English: {text}"

@app.route('/', methods=['GET', 'POST'])
def index():
    input_text = ''
    if request.method == 'POST':
        if 'translate' in request.form:
            input_text = request.form['input_text']
            lang = request.form['lang']
            output_text = pseudo_translate(input_text, lang)
            translation_history.clear()
            translation_history.append((input_text, output_text, lang))
        elif 'delete_history' in request.form:
            translation_history.clear()
            input_text = ''  # Clear input text when history is deleted

    return render_template('index.html', history=translation_history, input_text=input_text)

if __name__ == '__main__':
    app.run(debug=True)
