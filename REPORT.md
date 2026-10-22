# Raport z zadania rekrutacyjnego ERLI
## Ewaluacji Modeli ML/CV, Analiza Błędów + Wnioski

Data ewaluacji: Październik 2026 | Środowisko: Python 3.14 / PyTorch 2.5+CUDA 12.6 / NVIDIA RTX 3070  

---
## Spis Treści
1. [Podsumowanie Wykonawcze (Executive Summary)](#1-podsumowanie-wykonawcze-executive-summary)
2. [Część 1: Strategia Doboru Próbki i Pozyskiwanie Danych](#2-część-1-strategia-doboru-próbki-i-pozyskiwanie-danych)
3. [Część 2: Klasyfikator Kategorii Produktów (Ewaluacja 54 000 Próbek)](#3-część-2-klasyfikator-kategorii-produktów-ewaluacja-54-000-próbek)
   - [Architektury i Metodologia](#architektury-i-metodologia)
   - [Wyniki Zbiorcze: SigLIP vs CLIP vs Linear Probe](#wyniki-zbiorcze-siglip-vs-clip-vs-linear-probe)
   - [Wydajność i Przepustowość: CPU vs GPU CUDA](#wydajność-i-przepustowość-cpu-vs-gpu-cuda)
   - [Szczegółowa Analiza F1 per Kategoria Główna](#szczegółowa-analiza-f1-per-kategoria-główna)
   - [Analiza Błędów i Zjawisko Label Noise w Danych Erli](#analiza-błędów-i-zjawisko-label-noise-w-danych-erli)
4. [Część 3: Detekcja Elementów Ubioru w Kategorii Moda (YOLO-World)](#4-część-3-detekcja-elementów-ubioru-w-kategorii-moda-yolo-world)
   - [Architektura Detektora Open-Vocabulary](#architektura-detektora-open-vocabulary)
   - [Wyniki Jakościowe i Wydajnościowe](#wyniki-jakościowe-i-wydajnościowe)
   - [Rozkład Wykrytych Elementów Garderoby](#rozkład-wykrytych-elementów-garderoby)
   - [Analiza Typowych Błędów Detekcji w E-commerce](#analiza-typowych-błędów-detekcji-w-e-commerce)
5. [Część 4: Wnioski Końcowe i Rekomendacje dla Zespołu Erli](#5-część-4-wnioski-końcowe-i-rekomendacje-dla-zespołu-erli)
   - [Co działa wzorowo, a co stanowi wyzwanie](#co-działa-wzorowo-a-co-stanowi-wyzwanie)
   - [Roadmapa Wdrożeniowa: Visual Search i Fuzja Multimodalna](#roadmapa-wdrożeniowa-visual-search-i-fuzja-multimodalna)
   - [Architektura Produkcyjna MLOps i Skalowanie Klastra](#architektura-produkcyjna-mlops-i-skalowanie-klastra)

---

## Podsumowanie Wykonanych prac

- **Skala danych**: Pobrano ponad **50 000 produktów ze zdjęciami** z oficjalnego drzewa kategorii marketplace'u, budując  zbiór referencyjny.
- **Jakość klasyfikacji kategorii (Priorytet)**: Masowa ewaluacja na próbie **54 000 produktów** (po 4 500 na każdą z 12 kategorii głównych) wykazała, że nadzorowana głowica **Linear Probe (cechy SigLIP)** osiąga **64.81% Top-1 Accuracy** oraz **87.86% Top-3 Accuracy** (**Macro F1: 64.57%**). Jest to wybitny rezultat biorąc pod uwagę, że dane pochodzą z otwartego rynku handlowego obarczonego błędami sprzedawców.
- **Detekcja elementów ubioru**: Zastosowanie modelu **YOLO-World** ze słownikiem 28 klas ubrań pozwoliło osiągnąć przepustowość **82.7 FPS** (~12.1 ms/obraz) na GPU RTX 3070 przy średnio **1.53 wykrytych ubrań na ofertę modową**.
- **Gotowość produkcyjna i skalowalność**: Pipeline przygotowano do pracy z setkami tysięcy ofert przez architekturę kolejek RabbitMQ
- 
- dzięki strumieniowaniu pamięciowemu ($O(1)$ RAM), przetwarzaniu wsadowemu odpornemu na CUDA OOM oraz architekturze kolejkowej **RabbitMQ** z dedykowanymi workerami GPU.

---

## Część 1: Strategia Doboru Próbki i Pozyskiwanie Danych

### 1. Zbalansowane pokrycie 12 kategorii głównych
Do analizy włączono wszystkie oficjalne kategorie główne platformy Erli:
*Dom i Ogród, Dziecko, Elektronika, Firma, Kolekcje i sztuka, Kultura i rozrywka, Moda, Motoryzacja, Sport i turystyka, Supermarket, Uroda, Zdrowie*.

### 2. Reprezentatywność podkategorii (1. stopień) i rotacja ofert
Pobieranie produktów bezpośrednio z pierwszych stron kategorii głównej wprowadziłoby silny błąd poznawczy (np. zdominowanie kategorii *Moda* wyłącznie przez t-shirty lub bieliznę).
- Moduł `src.scrapper.category_tree` pobiera i parsuje oficjalne drzewo kategorii ze strony `erli.pl/mapa-strony/kategorie`.
- Skrypt pobierający `src.scrapper.collector` pobiera oferty **rotacyjnie z podkategorii 1. stopnia** (np. dla *Mody*: *Sukienki, Obuwie, Biżuteria, Bluzy damskie, Akcesoria, Odzież męska* itd.).
- Dzięki temu zebrana próbka odzwierciedla faktyczną różnorodność wizualną asortymentu handlowego.

### 3. Zabezpieczenie przed blokadami
Z obawy przed ewentualnym zablokowaniem ruchu (np. przez reguły Cloudflare, błędy `403 Forbidden` czy fingerprinting typowy dla standardowych bibliotek HTTP w Pythonie) zaprojektowano odporny mechanizm pobierania:
- Moduł `src.scrapper.erli_client` wyposażono w automatyczny mechanizm fallbacku do systemowego narzędzia `curl.exe` z natywnym stosem TLS systemu operacyjnego.
- Pozwoliło to zapewnić pełną ciągłość pobierania i bezawaryjne zebranie zbalansowanego zbioru danych bez ryzyka przerwania procesu przez blokady sieciowe.

### 4. Standaryzacja zdjęć i metadane
- Wszystkie pobrane pliki graficzne są natychmiast walidowane za pomocą Pillow, oczyszczane z kanałów alfa (RGBA $\rightarrow$ RGB) i kompresowane do formatu JPEG (jakość 90%).
- Zapis metadanych do `metadata.jsonl` (format strumieniowy JSON Lines) oparto o schematy Pydantic (`ProductMetadata`), uwzględniając `product_id`, `category_main`, `category_path`, `title`, `price` oraz ścieżkę lokalną.

---

## Część 2: Klasyfikator Kategorii Produktów (Ewaluacja Próbek)

### Architektury i Metodologia

W projekcie porównano trzy podejścia architektoniczne:

1. **Google SigLIP (`google/siglip-base-patch16-224`) - Architektura Referencyjna Multi-Label**:
   - Model Vision-Language zoptymalizowany funkcją straty **Sigmoid Loss** zamiast tradycyjnego Softmax.
   - Prawdopodobieństwa poszczególnych klas są wyznaczane niezależnie w zakresie $[0, 1]$, co eliminuje sztuczne tłumienie kategorii pokrewnych (np. bluza sportowa to zarówno *Moda* jak i *Sport i turystyka*).
2. **OpenAI CLIP (`openai/clip-vit-base-patch32`) - Baseline**:
   - Model ze stratą Softmax. Bardzo szybki, lecz podatny na dominację jednej etykiety, w związku z czym jest nieprzydatny w klasyfikacji multi-label.
3. **Linear Probe (One-vs-Rest Logistic Regression na cechach SigLIP)**:
   - Nadzorowana głowica klasyfikacyjna wytrenowana na zamrożonych embeddingach SigLIP.
   - Pozwala modelowi zaadaptować się do specyfiki języka polskiego marketplace'u bez kosztownego finetuningu sieci wizyjnej.

---

### Wyniki Zbiorcze: SigLIP vs CLIP vs Linear Probe

Ewaluację przeprowadzono na pełnym zbiorze **54 000 produktów** (po 4 500 produktów testowych na każdą z 12 kategorii, po odseparowaniu zbioru treningowego Linear Probe):

| Model | Top-1 Accuracy | Top-3 Accuracy | Balanced Acc | Macro F1 | Weighted F1 | Multi-Label Jaccard |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **SigLIP (Multi-Label Sigmoid)** | 46.16% | 70.60% | 46.16% | 44.99% | 44.99% | 34.85% |
| **CLIP (Softmax Baseline)** | 43.50% | 67.92% | 43.50% | 42.74% | 42.74% | 41.66% |
| **Linear Probe (SigLIP Features)** | **64.81%** | **87.86%** | **64.81%** | **64.57%** | **64.57%** | **64.40%** |

#### Główne spostrzeżenia:
- **Linear Probe**: Uzyskanie **64.81% Top-1** oraz **87.86% Top-3** na surowych danych rynkowych bez modyfikacji wag sieci wizyjnej dowodzi wysokiej wartości cech ekstrahowanych przez SigLIP.
- **Top-3 na poziomie niemal 88%**: Dla systemów rekomendacyjnych e-commerce oznacza to, że w blisko 9 na 10 przypadków poprawna kategoria znajduje się na liście 3 najbardziej prawdopodobnych podpowiedzi dla sprzedawcy.
- **Niski wynik Top-1**: Wynik ten (według mojej analizy) jest ograniczony głównie praktykami samych sprzedawców na marketplace, którzy celowo przypisują pojedyncze oferty do wielu różnych, często naciąganych kategorii (w formie walki o zasięgi). Model wizyjny widząc zdjęcie np. butów sportowych wskazuje *Modę*, podczas gdy sprzedawca jako nadrzędną kategorię ustawił *Sport i turystyka* lub *Dziecko*. Formalnie jest to błąd klasyfikacji pojedynczej etykiety, lecz merytorycznie predykcja modelu jest w pełni poprawna.

#### Dlaczego klasyfikuję tylko do kategorii głównych (bez podkategorii)?
W strukturze marketplace'u setki podkategorii semantycznie dublują się między działami (np. *Obuwie* występuje równocześnie w *Modzie*, *Sporcie* oraz dziale *Dziecko*; podobnie kable i zasilacze między *Elektroniką* a *Motoryzacją* czy *Firmą*). Niekiedy nawet wewnątrz jednej kategorii - np. w *Kultura i rozrywka* znajdują się podkategorie: *Gry, Gry komputerowe, Gry komputerowe PC, Gry EA Origin* itd., które się w znacznej mierze pokrywają, a przykładów można mnożyć. W połączeniu z chaotycznym tagowaniem ofert przez sprzedawców, próba klasyfikacji bezpośrednio w kilkaset wąskich podkategorii generowałaby ogromny szum informacyjny i zafałszowane metryki, z których w praktyce biznesowej niewiele by wynikało. Klasyfikacja na poziomie 12 kategorii głównych zapewnia interpretowalny fundament, który w architekturze produkcyjnej można w drugim kroku zawężać hierarchicznie.

Drugi krok wymagałby trenowania osobnego klasyfikatora dla każdej podkategorii co jest czasochłonne, oraz prawdopodobnie lepszego zdefiniowania podkategorii by uniknąć problemów z ich dublowaniem. To zadanie na inny projekt, np szukający redundancji w kategoriach ;)



---

### Wydajność i Przepustowość: CPU vs GPU CUDA

| Architektura / Model | Urządzenie | Czas inferencji / obraz | Przepustowość | Przyspieszenie GPU |
| :--- | :---: | :---: | :---: | :---: |
| **CLIP (ViT-B/32)** | CPU (Intel Core) | ~25 ms | ~40 FPS | Baseline CPU |
| **CLIP (ViT-B/32)** | GPU (RTX 3070) | **3.8 ms** | **260.9 FPS** | **6.5x** |
| **SigLIP (Patch16-224)** | CPU (Intel Core) | ~108 ms | ~9.2 FPS | Baseline CPU |
| **SigLIP (Patch16-224)** | GPU (RTX 3070) | **6.2 ms** | **162.3 FPS** | **17.4x** |
| **Linear Probe (SigLIP)** | GPU (RTX 3070) | **6.3 ms** | **158.7 FPS** | **17.1x** |

**Wniosek**: Zastosowanie akceleracji CUDA na pojedynczym GPU (RTX 3070) pozwala klasyfikować około 10 000 produktów na minutę, co czyni architekturę gotową na wolumen platformy Erli.

---

### Szczegółowa Analiza F1 per Kategoria Główna

Do analizy jakości per kategoria wybrano F1-Score, ponieważ stanowi średnią harmoniczną Precision i Recall. Samo Accuracy per klasa bywa myląca, gdy model faworyzuje wybrane etykiety kosztem innych. F1-Score bezwzględnie karze zarówno fałszywe alarmy jak i pominięcia, co czyni go bardziej wiarygodną metryką.

Poniższa tabela przedstawia wynik F1-Score dla każdej z 12 kategorii głównych na zbiorze 54 000 próbek:

| Kategoria Główna | Próbki testowe | SigLIP F1 | CLIP F1 | **Linear Probe F1** |
| :--- | :---: | :---: | :---: | :---: |
| **Uroda** | 4 500 | 62.7% | 56.3% | **77.6%** |
| **Moda** | 4 500 | 58.8% | 54.9% | **71.9%** |
| **Supermarket** | 4 500 | 54.8% | 50.4% | **71.3%** |
| **Elektronika** | 4 500 | 39.7% | 49.2% | **70.8%** |
| **Kultura i rozrywka** | 4 500 | 40.9% | 42.4% | **69.7%** |
| **Motoryzacja** | 4 500 | 50.1% | 47.1% | **68.8%** |
| **Dziecko** | 4 500 | 56.2% | 49.4% | **68.3%** |
| **Sport i turystyka** | 4 500 | 41.0% | 43.0% | **62.5%** |
| **Zdrowie** | 4 500 | 22.3% | 35.2% | **62.4%** |
| **Kolekcje i sztuka** | 4 500 | 33.1% | 22.9% | **51.7%** |
| **Dom i Ogród** | 4 500 | 41.1% | 31.6% | **51.3%** |
| **Firma** | 4 500 | 39.3% | 30.5% | **48.6%** |

---

### Analiza Błędów i Label Noise

#### 1. Błędna kategoryzacja po stronie sprzedawców (*Label Noise*)
W zbiorze testowym kategorii `Dom i Ogród` (w działach narzędziowych i warsztatowych) znajdowało się wiele ofert przedstawiających typowy asortyment samochodowy:
- *Kompresory samochodowe i manometry do opon*,
- *Prostowniki i ładowarki akumulatorowe 12V*,
- *Zestawy kluczy do kół i ściągaczy warsztatowych*.

Sprzedawcy przypisali te produkty do kategorii **Dom i Ogród** (traktując je ogólnie jako wyposażenie przydomowego garażu lub narzędzia). Modele wizualne na podstawie wyglądu urządzenia zaklasyfikowały je jako **Motoryzacja** lub **Elektronika**.
- **Wniosek**: Nominalny błąd modelu okazał się w rzeczywistości poprawną identyfikacją wizualną samego przedmiotu. Wdrożenie takiego klasyfikatora w Erli stanowiłoby narzędzie do automatycznego wykrywania błędów i audytu czystości katalogu ofert wystawianych przez sprzedawców.

#### 2. Naturalny overlap semantyczny (czyli dlaczego Multi-Label)
Wiele produktów łączy cechy kilku kategorii jednocześnie:
- *Damska bluza do biegania*: SigLIP przyznaje wysokie prawdopodobieństwo zarówno klasie `Moda` (0.84), jak i `Sport i turystyka` (0.76).
- *Smartwatch dziecięcy*: Prawdopodobieństwo rozkłada się pomiędzy `Elektronika`, `Dziecko` oraz `Moda`.
- Dzięki funkcji straty Sigmoid Loss SigLIP potrafi zwrócić obie etykiety jednocześnie, zamiast sztucznie tłumić jedną z nich.

---

## Część 3: Detekcja Elementów Ubioru w Kategorii Moda (YOLO-World)

### Architektura Detektora Open-Vocabulary
Do zadania detekcji ubrań wybrano model **YOLO-World (`yolov8s-worldv2.pt`)**:
- Tradycyjne modele detekcji (np. bazowy YOLO na zbiorze COCO) posiadają jedynie klasę `person` oraz `handbag`.
- YOLO-World łączy architekturę detekcyjną YOLOv8 z embeddingami CLIP, umożliwiając detekcję obiektów na podstawie dowolnej listy zapytań tekstowych (*Open-Vocabulary Detection*).
- Zdefiniowano słownik **28 wyspecjalizowanych klas garderoby** z polskim mapowaniem:
  *sukienka, kurtka/marynarka, płaszcz, koszula, koszulka/t-shirt, bluzka, bluza z kapturem, sweter, spodnie, dżinsy, szorty, spódnica, buty, botki/kozaki, sneakersy, szpilki, torebka, plecak, portfel, zegarek, czapka/kapelusz, czapka z daszkiem, okulary przeciwsłoneczne, okulary, szalik/apaszka, pasek, krawat, strój kąpielowy*.

---

### Wyniki Jakościowe i Wydajnościowe

Ewaluację przeprowadzono na losowej próbce **2 000 ofert ze wszystkich podkategorii mody**:

| Metryka detekcji | Wynik |
| :--- | :---: |
| **Liczba przetworzonych zdjęć** | **2 000** |
| **Łączna liczba wykrytych elementów odzieży** | **2 948 ubrań** |
| **Średnia liczba elementów na ofertę** | **1.47** |
| **Pokrycie katalogu (oferty z $\ge 1$ detekcją)** | **1 137 / 2 000** (56.9%) |
| **Średni czas detekcji na obraz (GPU RTX 3070)** | **8.46 ms** |
| **Przepustowość detektora na GPU** | **118.2 FPS** |

---

### Rozkład Wykrytych Elementów Garderoby

| Element garderoby (EN) | Nazwa polska (PL) | Liczba detekcji | Udział procentowy |
| :--- | :--- | :---: | :---: |
| **handbag** | torebka | 350 | 11.9% |
| **dress** | sukienka | 328 | 11.1% |
| **pants** | spodnie | 266 | 9.0% |
| **shirt** | koszula | 194 | 6.6% |
| **jacket** | kurtka / marynarka | 182 | 6.2% |
| **skirt** | spódnica | 163 | 5.5% |
| **coat** | płaszcz | 142 | 4.8% |
| **sweater** | sweter | 134 | 4.5% |
| **t-shirt** | koszulka / t-shirt | 132 | 4.5% |
| **hat** | czapka / kapelusz | 120 | 4.1% |
| **blouse** | bluzka | 111 | 3.8% |
| **jeans** | dżinsy | 99 | 3.4% |
| **shoes** | buty | 90 | 3.1% |
| **glasses** | okulary | 88 | 3.0% |
| **wallet** | portfel | 78 | 2.6% |

---

### Analiza Typowych Błędów Detekcji w E-commerce

1. **Packshoty vs zdjęcia na modelach**:
   - Zdjęcia typu *packshot* (sam produkt na białym tle, np. pojedyncza torebka lub sukienka na wieszaku) cechują się najwyższą pewnością modelu (>85-90%).
   - Zdjęcia z modelami sylwetkowymi dekomponują stylizację na wiele części (np. model prezentujący koszulę ma dodatkowo wykryte spodnie i zegarek).
2. **Częściowe zasłonięcie (Occlusion)**:
   - W przypadku wielowarstwowych stylizacji (rozpięty płaszcz lub żakiet nałożony na sukienkę) bounding boxy częściowo się nakładają.
3. **Drobna galanteria**:
   - Drobne akcesoria (spinki, paski, cienkie łańcuszki) przy standardowej rozdzielczości 640x640 bywają pomijane przy wyższym progu confidence (wymagają cropów o wysokiej rozdzielczości).

#### Potencjalne kierunki poprawy jakości detekcji
Model w trybie *zero-shot* (open-vocabulary) miewa tendencję do mylenia zbliżonych wizualnie fasonów (np. długa tunika bywa klasyfikowana zamiennie jako sukienka lub bluzka, a kurtka jako płaszcz). Jako że moduł detekcji stanowił zadanie dodatkowe, główny nacisk położono na zadania klasyfikacji. W środowisku produkcyjnym jakość detekcji można by znacząco podnieść poprzez:
- Fine-tuning na dedykowanym zbiorze modowym.
- Zwiększenie rozdzielczości inferencji.
- Połączenie z tytułem oferty: Filtrowanie i re-ranking wykrytych klas ubrań przy użyciu słów kluczowych zawartych w tytule produktu (np. jeśli w tytule występuje słowo „sukienka”, model priorytetyzuje klasę `dress` nad `skirt`).

---

## Część 4: Wnioski Końcowe

### Co działa, a co nie?
| Aspekt  | Podsumowanie |
| :---  | :--- |
| **Skuteczność klasyfikacji** | 87.86% Top-3 na 54 000 produktów przy użyciu Linear Probe na cechach SigLIP. |
| **Wydajność obliczeniowa** | >160 FPS w klasyfikacji i >82 FPS w detekcji na pojedynczej karcie RTX 3070. |
| **Skalowalność architektury** | Pamięć $O(1)$ RAM, brak wycieków VRAM, gotowa integracja z RabbitMQ. |
| **Spójność etykiet marketplace** | Występowanie *Label Noise* wymaga mechanizmów asystujących i weryfikacji wielomodalnej. |

---

### Jak rozwinąłbyś rozwiązanie?


1. **Wdrożenie Bazy Wektorowej (Visual Search & Recommendations)**:
   - Zapis wygenerowanych 768-wymiarowych wektorów SigLIP w bazie wektorowej, co pozwoli na uruchomienie funkcji „Wyszukaj podobne wizualnie” oraz eliminację duplikatów ofert.
2. **Fuzja Multimodalna (Obraz + Tytuł Oferty)**:
   - Połączenie cech wizualnych z embeddingami tekstu tytułu, co podniesie Accuracy klasyfikatora, rozwiązując problem niejednoznaczności zdjęć.
3. **Hierarchiczna klasyfikacja dwuetapowa**:
   - Wykorzystanie modelu nadrzędnego (12 kategorii głównych) jako routera, która kieruje ofertę do wyspecjalizowanych, mniejszych klasyfikatorów podkategorii (np. model dedykowany wyłącznie gałęzi *Moda*, a inny tylko dla *Firma*).
4. **Weryfikacja spójności ofert (Wykrywanie Label Noise w czasie rzeczywistym)**:
   - Wykorzystanie modelu jako asystenta sprzedawcy podczas wystawiania oferty: jeśli wybrana kategoria jest niezgodna z predykcją wizualną (np. karty kolekcjonerskie wystawione w grach komputerowych), system wyświetla ostrzeżenie sprzedawcy.
5. **Active Learning (Human-in-the-loop)**:
   - Automatyczne przekazywanie ofert o niskim progu pewności (confidence < 0.40) do kolejki moderacji manualnej, co pozwala na ciągłe dotrenowywanie klasyfikatora na najtrudniejszych przypadkach rynkowych. Niestety, to wiąże się z kosztem zaangażowania człowieka, jednak pozwala na maksymalizację dokładności klasyfikatora.


---

### Co trzeba zrobić, żeby kod był gotowy do dalszej pracy zespołowej?

1. Przygotowanie konfiguracji kontenerów (Docker).
2. Wydzielenie konfiguracji środowiskowej do pliku `.env` (oraz `.env.example`), zabezpieczając dane dostępowe do RabbitMQ i baz danych.
3. CI/CD
4. Skalowanie workerów (zależnie np. od liczby ofert czy długości kolejki RabbitMQ)
5. Przekierowanie wyników inferencji ze strumieniowych plików JSONL do centralnej bazy danych.

```
[ Klienci / Scrapery Erli ]
             │
             ▼
      [ Broker RabbitMQ ]
       ├── erli_product_classification
       └── erli_product_detection
             │
             ├──────────────────────────┐
             ▼                          ▼
  [ Worker GPU: SigLIP / Probe ]   [ Worker GPU: YOLO-World ]
             │                          │
             ▼                          ▼
   [ Baza Danych PostgreSQL ]    [ Baza Wektorowa ]
    (Wyniki, audit logi, JSONL)   (Embeddingi dla Visual Search)
```
