import glob
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

print("1. Veri setleri klasörden okunup birleştiriliyor...")
csv_dosyalari = glob.glob('veri_seti/*.csv')
df_listesi = []

for dosya in csv_dosyalari:
    try:
        df_gecici = pd.read_csv(dosya, on_bad_lines='skip', engine='python')
        df_gecici.columns = df_gecici.columns.str.lower()
        
        
        etiket_sutunu = None
        if 'label' in df_gecici.columns: etiket_sutunu = 'label'
        elif 'class' in df_gecici.columns: etiket_sutunu = 'class'
        
        
        metin_sutunu = None
        if 'text_combined' in df_gecici.columns: metin_sutunu = 'text_combined'
        elif 'text' in df_gecici.columns: metin_sutunu = 'text'
        elif 'body' in df_gecici.columns and 'subject' in df_gecici.columns:
            df_gecici['text_temp'] = df_gecici['subject'].astype(str) + " " + df_gecici['body'].astype(str)
            metin_sutunu = 'text_temp'
            
        if metin_sutunu and etiket_sutunu:
            df_final = df_gecici[[metin_sutunu, etiket_sutunu]].copy()
            df_final.columns = ['text', 'label']
            df_listesi.append(df_final)
    except Exception as e:
        print(f"Hata: {dosya} atlandı. {e}")


df = pd.concat(df_listesi, ignore_index=True)
df = df.dropna(subset=['text', 'label']) # Boş alanları burada temizleme işlemine sokuyorum
df['label'] = pd.to_numeric(df['label'], errors='coerce') # Sayı olmayanları NaN yaparak işlemlere devam ediyorum
df = df.dropna(subset=['label']) # Daha sonra NaN olanları temizliyorum
df['label'] = df['label'].astype(int) 

print(f"Toplam {len(df)} satır veri eğitiliyor...")

print("2. Model eğitiliyor...")
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
X = vectorizer.fit_transform(df['text'])
y = df['label']

model = MultinomialNB()
model.fit(X, y)

print("3. Model ve Vectorizer kaydediliyor...")
joblib.dump(model, 'phishing_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("İşlem başarıyla tamamlandı!")