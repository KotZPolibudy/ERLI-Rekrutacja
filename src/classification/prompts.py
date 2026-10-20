"""Szablony promptów semantycznych dla modeli Vision-Language (SigLIP, CLIP).
Opisy zawierają słowa kluczowe reprezentatywnych podkategorii, co diametralnie zwiększa
precyzję klasyfikacji produktów e-commerce.
"""

from typing import Dict, List
from src.common.config import MAIN_CATEGORIES

# Angielskie deskryptory semantyczne (rekomendowane dla SigLIP / CLIP z uwagi na dominację EN w pretrainingu)
CATEGORY_DESCRIPTIONS_EN: Dict[str, str] = {
    "Dom i Ogród": "home and garden product, furniture, interior decoration, tools, construction, renovation, lighting",
    "Dziecko": "baby and kids product, toys, stroller, car seat, baby care, feeding, children clothes",
    "Elektronika": "consumer electronics, smartphone, computer, laptop, TV, audio, photography, smart devices, cables",
    "Firma": "business, office supplies, packaging materials, warehouse equipment, professional signage, store supplies",
    "Kolekcje i sztuka": "collectibles, antiques, coins, stamps, art, paintings, vintage items, handicraft",
    "Kultura i rozrywka": "books, board games, musical instruments, vinyl records, movies, video games, comics",
    "Moda": "fashion clothing, shoes, apparel, dress, jacket, pants, jewelry, watches, handbags, accessories",
    "Motoryzacja": "car parts, automotive accessories, tires, motor oil, motorcycle gear, workshop tools, vehicle equipment",
    "Sport i turystyka": "sports equipment, fitness, camping, outdoor gear, bicycle, hiking, sportswear, training accessories",
    "Supermarket": "grocery, food, beverages, snacks, pet food, household cleaning supplies, laundry detergents",
    "Uroda": "beauty products, cosmetics, makeup, skincare, perfumes, hair care, nail care, manicure, personal grooming",
    "Zdrowie": "health and medical supplies, vitamins, dietary supplements, first aid, thermometers, pharmacy, rehabilitation equipment",
}

# Polskie deskryptory semantyczne
CATEGORY_DESCRIPTIONS_PL: Dict[str, str] = {
    "Dom i Ogród": "produkt do domu i ogrodu, meble, dekoracje wnętrz, narzędzia, budowa, oświetlenie",
    "Dziecko": "artykuły dla dzieci i niemowląt, zabawki, wózki, foteliki, klocki, pielęgnacja dziecka",
    "Elektronika": "sprzęt elektroniczny, smartfon, komputer, laptop, telewizor, audio, fotografia, akcesoria gsm",
    "Firma": "wyposażenie biura i firmy, materiały do pakowania, sprzęt magazynowy, reklama, artykuły biurowe",
    "Kolekcje i sztuka": "kolekcje, antyki, sztuka, monety, znaczki, rękodzieło, obrazy, pamiątki",
    "Kultura i rozrywka": "książki, gry planszowe, instrumenty muzyczne, płyty, filmy, gry wideo, komiksy",
    "Moda": "odzież, ubrania, buty, sukienki, kurtki, biżuteria, zegarki, torebki, akcesoria modowe",
    "Motoryzacja": "części samochodowe, akcesoria do auta, opony, oleje silnikowe, chemia motoryzacyjna, warsztat",
    "Sport i turystyka": "sprzęt sportowy, fitness, turystyka, kemping, rowery, wspinaczka, odzież sportowa",
    "Supermarket": "artykuły spożywcze, jedzenie, napoje, karma dla zwierząt, środki czystości, chemia domowa",
    "Uroda": "kosmetyki, makijaż, pielęgnacja twarzy i ciała, perfumy, pielęgnacja włosów, paznokcie",
    "Zdrowie": "zdrowie, apteczka, witaminy, suplementy diety, sprzęt medyczny, termometry, rehabilitacja",
}


def get_category_prompts(lang: str = "en", use_descriptions: bool = True) -> List[str]:
    """Zwraca listę promptów odpowiadającą kolejności MAIN_CATEGORIES.
    
    Args:
        lang: 'en' lub 'pl'
        use_descriptions: jeśli True, używa bogatych opisów semantycznych; jeśli False, tylko nazwy.
    """
    desc_dict = CATEGORY_DESCRIPTIONS_EN if lang == "en" else CATEGORY_DESCRIPTIONS_PL
    prompts: List[str] = []
    
    for cat in MAIN_CATEGORIES:
        if use_descriptions and cat in desc_dict:
            prompts.append(f"a photo of {desc_dict[cat]}" if lang == "en" else f"zdjęcie produktu: {desc_dict[cat]}")
        else:
            prompts.append(f"a photo of a {cat} product" if lang == "en" else f"zdjęcie produktu z kategorii {cat}")
            
    return prompts
