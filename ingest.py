import os
import requests
import json


# PDF'te zorunlu olanlar + 20'ye tamamlamak için eklenenler
PEOPLE = [
    "Albert Einstein", "Marie Curie", "Leonardo da Vinci", "William Shakespeare", 
    "Ada Lovelace", "Nikola Tesla", "Lionel Messi", "Cristiano Ronaldo", 
    "Taylor Swift", "Frida Kahlo", 
    # Eklenenler
    "Mustafa Kemal Atatürk", "İsmet İnönü", "Fevzi Çakmak", "Kazım Karabekir",
    "Sabiha Gökçen", "Ali Fuat Cebesoy", "Halide Edip Adıvar",
    "Enver Paşa", "Cemal Paşa", "Mithat Paşa", "Kanye West"
]

PLACES = [
    "Eiffel Tower", "Great Wall of China", "Taj Mahal", "Grand Canyon", 
    "Machu Picchu", "Colosseum", "Hagia Sophia", "Statue of Liberty", 
    "Giza Necropolis", "Mount Everest", # Pyramids of Giza -> Giza Necropolis (Wiki title)
    # Eklenenler
    "Dumlupınar", "Anıtkabir", "Topkapı Sarayı", "Galata Kulesi",
    "Çanakkale", "Kocatepe", "Eskişehir",
    "Tuna Nehri", "Ümraniye", "Hatay", "Şişli", "Selanik", "Vienna", "Ankara", "İstanbul"
]

def fetch_wikipedia_text(title):
    """Wikipedia API'sine istek atıp sayfanın düz metnini döndürür. Önce İngilizce, bulamazsa Türkçe Vikipedi'ye bakar."""
    headers = {
        "User-Agent": "LocalRAG/1.0 (local-rag-project; user@example.com)"
    }
    
    for lang in ["en", "tr"]:
        wiki_api_url = f"https://{lang}.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "prop": "extracts",
            "explaintext": "1",
            "titles": title,
            "redirects": "1",
            "format": "json"
        }
        
        try:
            response = requests.get(wiki_api_url, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            pages = data.get("query", {}).get("pages", {})
            for page_id, page_info in pages.items():
                if page_id != "-1" and page_info.get("extract"):
                    return page_info.get("extract", "")
                    
        except Exception as e:
            print(f"Hata! '{title}' ({lang} wiki) çekilirken sorun oluştu: {e}")

    print(f"Uyarı: '{title}' adında bir sayfa TR veya EN Wikipedia'da bulunamadı veya boş.")
    return None

def main():
    # Verileri kaydedeceğimiz klasörü oluştur
    os.makedirs("data", exist_ok=True)
    
    entities = {"person": PEOPLE, "place": PLACES}
    
    for entity_type, names in entities.items():
        print(f"--- {entity_type.upper()} verileri çekiliyor ---")
        for name in names:
            print(f"Çekiliyor: {name}...")
            text = fetch_wikipedia_text(name)
            
            if text:
                # Dosya adını temizle (Örn: "Albert Einstein" -> "albert_einstein")
                safe_name = name.lower().replace(" ", "_")
                filename = f"data/{entity_type}_{safe_name}.txt"
                
                # Metni utf-8 formatında txt olarak kaydet
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(text)
                
    print("\n[OK] Tüm veriler başarıyla çekildi ve 'data' klasörüne kaydedildi!")

if __name__ == "__main__":
    main()