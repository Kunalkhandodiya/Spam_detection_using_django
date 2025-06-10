from django.shortcuts import render
import joblib

def home(request):
    result =None
    if request.method == "GET":
        text = request.GET.get("text")
        if text:  # only process if text is entered
            vectorizer = joblib.load("C:/Users/Kunal/Downloads/spam_or_not/pickle_model/vectorizer.pkl")
            model = joblib.load("C:/Users/Kunal/Downloads/spam_or_not/pickle_model/model.pkl")
            termfreq = vectorizer.transform([text]).toarray()
            prediction = model.predict(termfreq)
            if prediction==0:
                result="ham"
            else:
                result="spam"
            return render(request, "app/templates/index.html", {"result": result})
    return render(request, "index.html")
