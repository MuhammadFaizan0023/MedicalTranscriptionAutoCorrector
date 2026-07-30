from flask import Flask, request, jsonify
from pipeline import AutocorrectPipeline
import nltk

# Ensure the sentence tokenizer model is available
nltk.download('punkt_tab', quiet=True)

app = Flask(__name__)

# Loads vocab + SymSpell + BERT once at startup (this can take a little while, especially first run)
print("Initializing pipeline...")
vocab_set = "./Data Files/merged_vocab_updated.pkl"
pipeline = AutocorrectPipeline(vocab_freq_path="./Data Files/merged_vocab_updated.pkl")
print("Pipeline initialized. Backend ready to accept requests.")


@app.route("/health", methods=["GET"])
def health():
    """Simple check to confirm the backend and model are up."""
    return jsonify({"status": "ok", "model": pipeline.model_name})


@app.route("/correct", methods=["POST"])
def correct():
    """
    Expects JSON: {"text": "a full document, one or more sentences, with typos"}
    Splits into sentences, corrects each independently, rejoins.
    Returns JSON: {"input": ..., "corrected": ..., "sentence_count": ...}
    """
    data = request.get_json(silent=True)

    if not data or "text" not in data:
        return jsonify({"error": "Request must include a 'text' field"}), 400

    text = data["text"].strip()

    if not text:
        return jsonify({"error": "Text cannot be empty"}), 400

    try:
        sentences = nltk.sent_tokenize(text)
        corrected_sentences = [pipeline.correct_sentence(s) for s in sentences]
        corrected = " ".join(corrected_sentences)
    except Exception as e:
        return jsonify({"error": f"Correction failed: {str(e)}"}), 500

    return jsonify({
        "input": text,
        "corrected": corrected,
        "sentence_count": len(sentences)
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)