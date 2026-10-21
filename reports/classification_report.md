# Raport Porównawczy Modeli Klasyfikacji Kategorii Produktów (ERLI)

> Wygenerowano automatycznie w ramach potoku ewaluacji modeli ML/CV.

## 1. Zbiorcze zestawienie metryk jakościowych i wydajnościowych

| Model | Top-1 Accuracy | Top-3 Accuracy | Balanced Acc | Macro F1 | Weighted F1 | Multi-Label Jaccard | Latencja (CPU) | Przepustowość |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SigLIP (Multi-Label Sigmoid)** | 46.16% | 70.60% | 46.16% | 44.99% | 44.99% | 34.85% | 6.2 ms | 160.2 FPS |
| **CLIP (Softmax Baseline)** | 43.50% | 67.92% | 43.50% | 42.74% | 42.74% | 41.66% | 3.0 ms | 328.1 FPS |
| **Linear Probe (SigLIP Features)** | 64.81% | 87.86% | 64.81% | 64.57% | 64.57% | 64.40% | 6.4 ms | 157.2 FPS |

## 2. Szczegółowe metryki F1-Score per kategoria główna

| Kategoria | Liczba próbek | **SigLIP (Multi-Label Sigmoid)** F1 | **CLIP (Softmax Baseline)** F1 | **Linear Probe (SigLIP Features)** F1 |
| :--- | :---: | :---: | :---: | :---: |
| Dom i Ogród | 4500 | 41.1% | 31.6% | 51.3% |
| Dziecko | 4500 | 56.2% | 49.4% | 68.3% |
| Elektronika | 4500 | 39.7% | 49.2% | 70.8% |
| Firma | 4500 | 39.3% | 30.5% | 48.6% |
| Kolekcje i sztuka | 4500 | 33.1% | 22.9% | 51.7% |
| Kultura i rozrywka | 4500 | 40.9% | 42.4% | 69.7% |
| Moda | 4500 | 58.8% | 54.9% | 71.9% |
| Motoryzacja | 4500 | 50.1% | 47.1% | 68.8% |
| Sport i turystyka | 4500 | 41.0% | 43.0% | 62.5% |
| Supermarket | 4500 | 54.8% | 50.4% | 71.3% |
| Uroda | 4500 | 62.7% | 56.3% | 77.6% |
| Zdrowie | 4500 | 22.3% | 35.2% | 62.4% |

## 3. Kluczowe wnioski biznesowe i architektoniczne

- **Wsparcie dla Multi-Label**: Google SigLIP dzięki niezależnej stracie sigmoid loss pozwala przypisać produkt do więcej niż jednej kategorii (np. odzież sportowa = *Moda* + *Sport i turystyka*), podczas gdy OpenAI CLIP wymusza sumę prawdopodobieństw do 1 (softmax), sztucznie tłumiąc kategorie pokrewne.
- **Wydajność na CPU**: Zarówno SigLIP jak i CLIP osiągają średni czas inferencji na poziomie poniżej 100-110 ms/obraz na standardowym CPU, co przekłada się na ~10-12 FPS bez potrzeby drogich kart graficznych.
- **Linear Probe**: Wytrenowanie lekkiej głowicy wieloetykietowej (One-vs-Rest) na zamrożonych embeddingach SigLIP umożliwia adaptację do specyfiki nazewnictwa w serwisie Erli bez kosztownego finetuningu wag wizyjnych.
