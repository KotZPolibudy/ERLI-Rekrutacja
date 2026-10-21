# Pipeline ML/CV dla platformy E-commerce ERLI
## Instrukcja Obsługi projektu / zadania rekrutacyjnego

**Polecam zacząć od przeczytania raportu, przed uruchamianiem projektu:**  
 **[REPORT.md](REPORT.md)**

---

## Spis Treści
- [1. Wymagania Sprzętowe i Programowe](#1-wymagania-sprzętowe-i-programowe)
- [2. Odtworzenie Środowiska Krok po Kroku](#2-odtworzenie-środowiska-krok-po-kroku)
- [3. Instrukcja Uruchomienia Poszczególnych Modułów](#3-instrukcja-uruchomienia-poszczególnych-modułów)
  - [Krok 1: Pobieranie Próbki Produktów (Scraper)](#krok-1-pobieranie-próbki-produktów-scraper)
  - [Krok 2: Klasyfikacja Kategorii Produktów](#krok-2-klasyfikacja-kategorii-produktów)
  - [Krok 3: Detekcja Elementów Ubioru (Kategoria Moda)](#krok-3-detekcja-elementów-ubioru-kategoria-moda)
  - [Krok 4: Skalowanie (RabbitMQ + GPU)](#krok-4-skalowanie-rabbitmq--gpu)
- [4. Testy Jednostkowe](#4-testy-jednostkowe)
- [5. Struktura Katalogów Projektu](#5-struktura-katalogów-projektu)
- [6. Deklaracja Użycia Narzędzi AI](#6-deklaracja-użycia-narzędzi-ai)

---

## 1. Wymagania Sprzętowe i Programowe

### Wymagania bazowe:
- **Python**: wersja **>= 3.10** (zalecana 3.11 – 3.14).
- **System operacyjny**: Windows 10/11 lub Linux (Ubuntu 20.04+).
- **Docker** *(opcjonalnie)*: wymagany jedynie w przypadku uruchamiania kolejki RabbitMQ.

### Sprzęt obliczeniowy:
- **Karta graficzna GPU (Zalecane)**: dedykowana karta **NVIDIA** ze sterownikami wspierającymi **CUDA 12.6+** (np. seria GeForce RTX 30xx / 40xx lub GTX) – zapewnia ~15x szybsze przetwarzanie (>160 FPS w klasyfikacji, >80 FPS w detekcji).
- **Procesor CPU (Wspierane)**: w przypadku braku karty NVIDIA pipeline automatycznie przełącza się w tryb CPU (pełna funkcjonalność i identyczne wyniki, bez konieczności konfiguracji sterowników).

---

## 2. Odtworzenie Środowiska Krok po Kroku

### Krok 2.1: Sklonowanie repozytorium
```bash
git clone https://github.com/KotZPolibudy/ERLI-Rekrutacja.git
cd ERLI-Rekrutacja
```

### Krok 2.2: Utworzenie i aktywacja środowiska wirtualnego
- **Windows (PowerShell):**
  ```powershell
  python -m venv .venv
  . .venv\Scripts\Activate.ps1
  ```
- **Linux / macOS:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

### Krok 2.3: Instalacja zależności

- **Opcja A: Z akceleracją GPU CUDA (Rekomendowane):**
  ```powershell
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
  pip install -r requirements.txt
  ```

- **Opcja B: Tylko CPU:**
  ```powershell
  pip install -r requirements.txt
  ```

---

## 3. Instrukcja Uruchomienia Poszczególnych Modułów

### Krok 1: Pobieranie Próbki Produktów (Scraper)

Moduł pobiera oferty rotacyjnie z podkategorii 1. stopnia wszystkich 12 kategorii głównych, weryfikuje integralność obrazów i zapisuje metadane do `data/raw/metadata.jsonl`:

```powershell
# Pobranie np. 300 produktów na kategorię główną (300 x 12 kategorii = 3600 produktów)
python -m src.scrapper.collector --target-per-category 300
```

- **Wznawianie**: Proces można w dowolnym momencie zatrzymać (`Ctrl+C`) i wznowić bez duplikacji danych.
- **Lokalizacja danych**:
  - Zdjęcia: `data/raw/<Kategoria_Główna>/<Podkategoria>/<product_id>.jpg`
  - Metadane: `data/raw/metadata.jsonl`

---

### Krok 2: Klasyfikacja Kategorii Produktów

#### A. Uruchomienie benchmarku i ewaluacji modeli:
Skrypt `src.evaluation.benchmark` umożliwia dobór modeli oraz wielkości próby testowej:

- **Ewaluacja porównawcza wszystkich 3 modeli (SigLIP, CLIP, Linear Probe):**
  ```powershell
  python -m src.evaluation.benchmark --all-models --device cuda
  ```
- **Ewaluacja jednego wybranego modelu (np. tylko SigLIP):**
  ```powershell
  python -m src.evaluation.benchmark --single-model siglip --device cuda
  ```
- **Własny dobór wielkości zbioru testowego:**
  ```powershell
  python -m src.evaluation.benchmark --all-models --sample-ratio 0.1 --ratio 0.25 --device cuda
  ```
- **Ewaluacja pełnej bazy produktów (np. 50 000+ ofert):**
  ```powershell
  python -m src.evaluation.benchmark --all-models --all-samples --ratio 0.25 --device cuda
  ```

Wyniki, macierze pomyłek PNG oraz pliki analizy błędów zapisywane są w unikalnym podfolderze `reports/benchmark_{device}_{timestamp}/` oraz zbiorczo w `reports/`.

#### B. Klasyfikacja pojedynczego zdjęcia:
```powershell
python -m src.classification.predict --model siglip --image "data/samples/images/Moda/sample_001.jpg"
```
Zwraca kategorię główną, pewność, predykcje wieloetykietowe (multi-label) oraz Top-3.

#### C. Masowa klasyfikacja zbioru danych (Batch Inference z pamięcią $O(1)$ RAM):
```powershell
python -m src.classification.predict --model siglip --metadata data/raw/metadata.jsonl --batch-size 64
```

---

### Krok 3: Detekcja Elementów Ubioru (Kategoria Moda)

Moduł detekcji `src.detection` wykorzystuje model **YOLO-World** ze słownikiem 28 klas ubrań i akcesoriów.

#### A. Uruchomienie benchmarku detekcji:
```powershell
# Ewaluacja na 100 ofertach z kategorii Moda wraz z wygenerowaniem mozaiki wizualnej
python -m src.detection.benchmark_detection --device cuda --samples 100 --visualize-count 12
```

- **Generowane pliki**:
  - Raport Markdown: `reports/fashion_detection_report.md`
  - Mozaika przykładowych detekcji: `reports/fashion_detection_mosaic.jpg`
  - Pojedyncze obrazy z naniesionymi ramkami i etykietami: `reports/detection_cuda_*/samples/`

---

### Krok 4: Skalowanie (RabbitMQ + GPU)

Architektura oparta o brokera wiadomości i workerów GPU pozwala na łatwe skalowanie horyzontalne przetwarzania setek tysięcy ofert w czasie rzeczywistym.


**Wymagania wstępne:** Upewnij się, że aplikacja **Docker Desktop** jest uruchomiona (silnik Docker Engine musi mieć aktywny status). Do uruchomienia przetwarzania kolejkowego potrzebne będą **dwa osobne okna terminala** (jedno dla workera w pętli ciągłej, drugie dla producenta).

#### 1. Uruchomienie brokera RabbitMQ (Docker Compose):
```powershell
docker compose up -d
```
Panel webowy zarządzania kolejkami dostępny jest pod adresem: `http://localhost:15672` (login: `guest`, hasło: `guest`).

#### 2. Uruchomienie konsumenta / workera GPU (Terminal 1):
Worker przechodzi w stan ciągłego nasłuchiwania i natychmiastowo przetwarza spływające paczki ofert (GPU batching):
- **Dla klasyfikacji kategorii (SigLIP / CLIP):**
  ```powershell
  python -m src.queue.worker --model siglip --device cuda --batch-size 64
  ```
- **Dla detekcji elementów ubioru:**
  ```powershell
  python -m src.queue.detection_worker --device cuda --batch-size 32
  ```

#### 3. Zasilenie kolejki zadaniami (Terminal 2 - Producent):
W nowym oknie konsoli uruchom producenta, który opublikuje zadania do kolejki:
- **Wysyłanie produktów do klasyfikacji kategorii:**
  ```powershell
  python -m src.queue.producer --limit 500
  ```
- **Wysyłanie wyłącznie ofert z kategorii Moda do detektora ubrań:**
  ```powershell
  python -m src.queue.producer --fashion-only --limit 500
  ```

---

## 4. Testy Jednostkowe

Zestaw 16 testów jednostkowych weryfikuje poprawność schematów Pydantic, klienta pobierającego dane z Erli, datasetu PyTorch, modułu metryk ewaluacyjnych oraz detektora modowego:

```powershell
pytest
```

Oczekiwany rezultat: **16 passed in ~20s (100% sukcesu)**.

---

## 5. Struktura Katalogów Projektu

```
ERLI-Rekrut/
├── data/
│   ├── raw/                                 # Pobierane zdjęcia pogrupowane według kategorii
│   │   ├── Dom i Ogród/
│   │   ├── Moda/
│   │   │   └── Bluzy damskie/
│   │   └── metadata.jsonl                   # Główny plik metadanych produktów
│   ├── samples/                             # Zestaw mockowy (szybkie testy jednostkowe)
│   └── category_tree.json                   # Sparsowane oficjalne drzewo kategorii Erli
├── src/
│   ├── common/                              # Wspólne schematy Pydantic i konfiguracja
│   │   ├── config.py                        # Definicja 12 kategorii głównych i stałych ścieżek
│   │   └── schemas.py                       # Modele: ProductMetadata, CategoryPrediction, BBox
│   ├── scrapper/                            # Część 1: Moduł pobierania danych
│   │   ├── category_tree.py                 # Ekstraktor drzewa kategorii z mapy serwisu
│   │   ├── erli_client.py                   # Odporny klient HTTP (Cloudflare bypass via curl fallback)
│   │   └── collector.py                     # Skrypt pobierający z rotacją podkategorii
│   ├── classification/                      # Część 2: Klasyfikatory kategorii
│   │   ├── prompts.py                       # Bogate deskryptory semantyczne kategorii (EN i PL)
│   │   ├── model.py                         # SigLIP, CLIP oraz Linear Probe z pamięciowym batchingiem
│   │   ├── dataset.py                       # PyTorch ProductDataset ze stratyfikacją
│   │   └── predict.py                       # CLI inferencji pojedynczej i masowej
│   ├── detection/                           # Część 3: Detekcja elementów ubioru
│   │   ├── detector.py                      # FashionDetector (YOLO-World z 28 klasami ubrań)
│   │   ├── visualize.py                     # Rysowanie bounding boxów, paleta barw i mozaika
│   │   └── benchmark_detection.py           # Benchmark latencji, rozkład klas i raport Markdown
│   ├── queue/                               # Architektura rozproszona (RabbitMQ)
│   │   ├── config.py                        # Konfiguracja brokera AMQP
│   │   ├── producer.py                      # Publikowanie zadań z filtrem kategorii
│   │   ├── worker.py                        # Konsument GPU dla klasyfikacji
│   │   └── detection_worker.py              # Konsument GPU dla detekcji ubrań
│   └── evaluation/                          # Ewaluacja, metryki i benchmarki
│       ├── metrics.py                       # Metryki: Acc, Top-3, F1, Jaccard, Confusion Matrix
│       └── benchmark.py                     # Skrypt masowego benchmarku (GPU/CPU, podział ratio)
├── reports/                                 # Wygenerowane raporty ewaluacyjne, wykresy i macierze PNG
│   ├── classification_report.md             # Raport klasyfikatora (54k próbek)
│   ├── fashion_detection_report.md          # Raport detektora ubrań
│   ├── fashion_detection_mosaic.jpg         # Mozaika poglądowa detekcji ubrań
│   └── benchmark_cuda_*/                    # Historyczne unikalne uruchomienia benchmarków
├── tests/                                   # Zestaw 16 testów jednostkowych (pytest)
├── docker-compose.yml                       # Gotowe środowisko RabbitMQ
├── requirements.txt                         # Zależności projektu
├── REPORT.md                                # GŁÓWNY RAPORT EWALUACYJNY I WNIOSKI BIZNESOWE
├── PLAN.md                                  # Harmonogram i podział zadań na strumienie
├── Zadanie.md                               # Oryginalna treść zadania rekrutacyjnego
└── README.md                                # Niniejsza instrukcja uruchomienia
```

---

## 6. Deklaracja Użycia Narzędzi AI
- Narzędzia AI wykorzystano do:
  - Generowania powtarzalnego kodu pomocniczego (modele Pydantic, obsługa argumentów CLI, konfiguracja pytest, generowania mozaiki i temu podobne).
  - Wsparcia w doborze promptów semantycznych dla modeli Vision-Language.
  - Automatyzacji i optymalizacji zapytań I/O oraz konfiguracji środowiska CUDA 12.6.
  - Pisania tej instrukcji, tak aby była dobrze zrozumiała 
