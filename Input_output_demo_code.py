
#  Bangla Sentiment Prediction Loop


def predict_single_review(text):
    """
    Clean a raw Bangla review, tokenize it, pad it,
    and use the GOOD model to predict sentiment.
    """
    if not text or len(text.strip()) == 0:
        print("⚠️ Empty input. Please type a valid Bangla review.")
        return None

    # Clean using same function as training
    cleaned = clean_text_bangla(text)
    print(f"\n📌 Cleaned text: {cleaned}")

    # Convert to sequence & pad
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=maxlen, padding="post", truncating="post")

    # Predict sentiment using GOOD model
    model = models.get("good", good_model)
    prob = model.predict(padded, verbose=0)[0][0]

    # Threshold and label
    label = 1 if prob >= 0.5 else 0
    label_name = "😊 Positive (1)" if label == 1 else "😡 Negative/Neutral (0)"

    print(f"\n🔢 Prediction probability: {prob:.4f}")
    print(f"🎯 Predicted label: {label_name}")

    return prob, label


#  Continuous Prediction Mode

print("\n🔎 Bangla Movie Review Sentiment Detector is Ready!")
print("👉 Type a review to classify sentiment.")
print("❌ Type 'exit' to stop.\n")

while True:
    try:
        user_text = input("📝 Enter Bangla review: ").strip()
        if user_text.lower() == "exit":
            print("\n👋 Exiting prediction mode. Thank you!")
            break
        predict_single_review(user_text)
        print("\n----------------------------------------\n")
    except KeyboardInterrupt:
        print("\n⛔ Stopped by user.")
        break
    except EOFError:
        print("\n❗ Input ended.")
        break
