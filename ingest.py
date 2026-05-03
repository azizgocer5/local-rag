import os
import requests
import json


# Mandatory in PDF + added to complete to 20
PEOPLE = [
    "Albert Einstein", "Marie Curie", "Leonardo da Vinci", "William Shakespeare", 
    "Ada Lovelace", "Nikola Tesla", "Lionel Messi", "Cristiano Ronaldo", 
    "Taylor Swift", "Frida Kahlo", 
    # Added
    "Mustafa Kemal Atatürk", "İsmet İnönü", "Fevzi Çakmak", "Kazım Karabekir",
    "Sabiha Gökçen", "Ali Fuat Cebesoy", "Halide Edip Adıvar",
    "Enver Paşa", "Cemal Paşa", "Mithat Paşa", "Talat Paşa", "Kanye West"
]

PLACES = [
    "Eiffel Tower", "Great Wall of China", "Taj Mahal", "Grand Canyon", 
    "Machu Picchu", "Colosseum", "Hagia Sophia", "Statue of Liberty", 
    "Giza Necropolis", "Mount Everest", # Pyramids of Giza -> Giza Necropolis (Wiki title)
    # Added
    "Dumlupınar", "Anıtkabir", "Topkapı Sarayı", "Galata Kulesi",
    "Çanakkale", "Kocatepe", "Eskişehir",
    "Tuna Nehri", "Ümraniye", "Hatay", "Şişli", "Selanik", "Vienna", "Ankara", "İstanbul"
]

def fetch_wikipedia_text(title):
    """Sends a request to the Wikipedia API and returns the plain text of the page. It checks English Wikipedia first, then Turkish if not found."""
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
            print(f"Error! An issue occurred while fetching '{title}' ({lang} wiki): {e}")

    print(f"Warning: No page named '{title}' could be found on TR or EN Wikipedia, or it is empty.")
    return None

def main():
    # Create the directory where we will save the data
    os.makedirs("data", exist_ok=True)
    
    entities = {"person": PEOPLE, "place": PLACES}
    
    for entity_type, names in entities.items():
        print(f"--- Fetching {entity_type.upper()} data ---")
        for name in names:
            print(f"Fetching: {name}...")
            text = fetch_wikipedia_text(name)
            
            if text:
                # Clean the filename (e.g., "Albert Einstein" -> "albert_einstein")
                safe_name = name.lower().replace(" ", "_")
                filename = f"data/{entity_type}_{safe_name}.txt"
                
                # Save the text as a txt file in utf-8 format
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(text)
                
    print("\n[OK] All data successfully fetched and saved to the 'data' directory!")

if __name__ == "__main__":
    main()