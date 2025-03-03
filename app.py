from flask import Flask, render_template, request, redirect
from helper import analyze_text
from logger import logging

app = Flask(__name__)

logging.info('Flask server started')

# Store sentiment counts and reviews
data = {
    "reviews": [],
    "positive": 0,
    "negative": 0
}


@app.route("/")
def index():
    logging.info('========== Open home page ===========')
    return render_template('index.html', data=data)


@app.route("/", methods=['POST'])
def my_post():
    text = request.form['text'].strip()
    language = request.form.get('language', 'en')  # Get language from the form, default to 'en'

    if not text:
        logging.warning("Empty text received")
        return redirect(request.url)

    logging.info(f"Input Text: {text}")
    logging.info(f"Selected Language: {language}")

    try:
        # Pass the text and language to the helper function
        prediction = analyze_text(text, language)  # Process text and predict sentiment
        logging.info(f'➡️ Final Prediction: {prediction}')
    except Exception as e:
        logging.error(f"Error in sentiment analysis pipeline: {e}")
        return redirect(request.url)

    # Update sentiment counts
    if prediction == 'negative':
        data["negative"] += 1
    else:
        data["positive"] += 1

    # Store the review
    data["reviews"].insert(0, {
        "text": text,
        "sentiment": prediction
    })

    return redirect(request.url)  # Reload the page to update UI


if __name__ == "__main__":
    app.run(debug=True)
