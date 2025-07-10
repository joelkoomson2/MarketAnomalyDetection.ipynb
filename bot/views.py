import os
from django.shortcuts import render
from dotenv import load_dotenv
import joblib
import pandas as pd
import google.generativeai as genai

# Load env and configure Gemini API
load_dotenv()
genai.configure()  # Will auto-pick up GOOGLE_API_KEY from env

# Load trained model once
rf_model = joblib.load("rf_model.pkl")

def index(request):
    import os
    print("GOOGLE_API_KEY:", os.getenv("GOOGLE_API_KEY"))
    answer = ""
    question = ""
    vix = ""
    dxy = ""
    model_prob = ""

    if request.method == "POST":
        question = request.POST.get("question", "")
        vix = request.POST.get("vix", "")
        dxy = request.POST.get("dxy", "")

        # Validate & predict if inputs provided
        if vix and dxy:
            try:
                vix_value = float(vix)
                dxy_value = float(dxy)
                input_df = pd.DataFrame([[vix_value, dxy_value]], columns=['VIX', 'DXY'])
                prob = rf_model.predict_proba(input_df)[0][1]
                model_prob = f"The model predicts a crash probability of {prob:.2f}."
            except ValueError:
                model_prob = "Invalid input: please enter numbers for VIX and DXY."
        else:
            model_prob = "No input values provided for prediction."

        # Compose context for Gemini
        prompt = (
            f"You are an AI investment strategy assistant.\n"
            f"User question: {question}\n"
            f"Model insight: {model_prob}\n"
            f"Explain clearly for an investor: how does the model work, "
            f"what does this mean, and what should they do?"
        )

        # Use Google Gemini API with supported model
        model = genai.GenerativeModel('models/gemini-1.5-pro-latest')
        response = model.generate_content(prompt)
        answer = response.text.strip() if hasattr(response, 'text') else str(response)

    return render(request, 'index.html', {
        'question': question,
        'vix': vix,
        'dxy': dxy,
        'model_prob': model_prob,
        'response': answer
    })
