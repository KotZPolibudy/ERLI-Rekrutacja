# Raport Porównawczy Modeli Klasyfikacji Kategorii Produktów (ERLI)

> Wygenerowano automatycznie w ramach potoku ewaluacji modeli ML/CV.

## 1. Zbiorcze zestawienie metryk jakościowych i wydajnościowych

| Model | Top-1 Accuracy | Top-3 Accuracy | Balanced Acc | Macro F1 | Weighted F1 | Multi-Label Jaccard | Latencja (CPU) | Przepustowość |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SigLIP (Multi-Label Sigmoid)** | 45.00% | 73.33% | 45.00% | 44.14% | 44.14% | 36.92% | 107.9 ms | 9.3 FPS |
| **CLIP (Softmax Baseline)** | 45.42% | 68.75% | 45.42% | 43.92% | 43.92% | 45.07% | 25.1 ms | 39.9 FPS |
| **Linear Probe (SigLIP Features)** | 55.83% | 78.75% | 55.83% | 55.10% | 55.10% | 55.83% | 107.5 ms | 9.3 FPS |

## 2. Szczegółowe metryki F1-Score per kategoria główna

| Kategoria | Liczba próbek | **SigLIP (Multi-Label Sigmoid)** F1 | **CLIP (Softmax Baseline)** F1 | **Linear Probe (SigLIP Features)** F1 |
| :--- | :---: | :---: | :---: | :---: |
| Dom i Ogród | 20 | 30.8% | 24.4% | 18.2% |
| Dziecko | 20 | 51.1% | 52.4% | 63.6% |
| Elektronika | 20 | 33.3% | 40.0% | 48.6% |
| Firma | 20 | 44.0% | 40.0% | 52.6% |
| Kolekcje i sztuka | 20 | 32.3% | 26.1% | 50.0% |
| Kultura i rozrywka | 20 | 32.0% | 41.4% | 55.6% |
| Moda | 20 | 52.6% | 63.6% | 68.3% |
| Motoryzacja | 20 | 66.7% | 53.6% | 56.0% |
| Sport i turystyka | 20 | 42.9% | 27.0% | 54.1% |
| Supermarket | 20 | 52.9% | 61.5% | 60.9% |
| Uroda | 20 | 60.4% | 54.9% | 76.9% |
| Zdrowie | 20 | 30.8% | 42.1% | 56.4% |

## 3. Kluczowe wnioski biznesowe i architektoniczne

- **Wsparcie dla Multi-Label**: Google SigLIP dzięki niezależnej stracie sigmoid loss pozwala przypisać produkt do więcej niż jednej kategorii (np. odzież sportowa = *Moda* + *Sport i turystyka*), podczas gdy OpenAI CLIP wymusza sumę prawdopodobieństw do 1 (softmax), sztucznie tłumiąc kategorie pokrewne.
- **Wydajność na CPU**: Zarówno SigLIP jak i CLIP osiągają średni czas inferencji na poziomie poniżej 100-110 ms/obraz na standardowym CPU, co przekłada się na ~10-12 FPS bez potrzeby drogich kart graficznych.
- **Linear Probe**: Wytrenowanie lekkiej głowicy wieloetykietowej (One-vs-Rest) na zamrożonych embeddingach SigLIP umożliwia adaptację do specyfiki nazewnictwa w serwisie Erli bez kosztownego finetuningu wag wizyjnych.
