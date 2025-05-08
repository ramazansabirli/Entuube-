
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Model ve kodlayıcılar
with open("entubasyon_model.pkl", "rb") as f:
    model = pickle.load(f)

tip_encoder = LabelEncoder().fit(['tip1', 'tip2', 'mixt'])
cinsiyet_encoder = LabelEncoder().fit(['Erkek', 'Kadın'])
destek_encoder = LabelEncoder().fit(['Oda havası', 'Nazal oksijen', 'Maske', 'High Flow', 'NIMV'])
tedavi_classes = ['Entübasyon', 'NIMV', 'İzlem / Oksijen']

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
def form():
    return """
    <html>
        <body>
            <h2>Entübasyon Karar Desteği</h2>
            <form action="/predict" method="post">
                Tip: <input name="Tip" value="tip1"><br>
                Yaş: <input name="Yaş" type="number"><br>
                Cinsiyet: <input name="Cinsiyet" value="Erkek"><br>
                GCS: <input name="GCS" type="number"><br>
                SpO2: <input name="SpO2" type="number" step="0.1"><br>
                Solunum Sayısı: <input name="Solunum_Sayısı" type="number"><br>
                Kalp Hızı: <input name="Kalp_Hızı" type="number"><br>
                SBP: <input name="SBP" type="number"><br>
                DBP: <input name="DBP" type="number"><br>
                pH: <input name="pH" type="number" step="0.01"><br>
                PaCO2: <input name="PaCO2" type="number" step="0.1"><br>
                PaO2: <input name="PaO2" type="number" step="0.1"><br>
                FiO2: <input name="FiO2" type="number"><br>
                Mevcut Destek: <input name="Mevcut_Destek" value="Nazal oksijen"><br>
                <input type="submit">
            </form>
        </body>
    </html>
    """

@app.post("/predict", response_class=HTMLResponse)
def predict(
    Tip: str = Form(...),
    Yaş: int = Form(...),
    Cinsiyet: str = Form(...),
    GCS: int = Form(...),
    SpO2: float = Form(...),
    Solunum_Sayısı: int = Form(...),
    Kalp_Hızı: int = Form(...),
    SBP: int = Form(...),
    DBP: int = Form(...),
    pH: float = Form(...),
    PaCO2: float = Form(...),
    PaO2: float = Form(...),
    FiO2: int = Form(...),
    Mevcut_Destek: str = Form(...)
):
    df = pd.DataFrame([{
        "Tip": tip_encoder.transform([Tip])[0],
        "Yaş": Yaş,
        "Cinsiyet": cinsiyet_encoder.transform([Cinsiyet])[0],
        "GCS": GCS,
        "SpO2": SpO2,
        "Solunum Sayısı": Solunum_Sayısı,
        "Kalp Hızı": Kalp_Hızı,
        "SBP": SBP,
        "DBP": DBP,
        "pH": pH,
        "PaCO2": PaCO2,
        "PaO2": PaO2,
        "FiO2": FiO2,
        "Mevcut Destek": destek_encoder.transform([Mevcut_Destek])[0]
    }])
    
    prediction = model.predict(df)[0]
    return f"<h2>Önerilen Tedavi: {tedavi_classes[prediction]}</h2>"
