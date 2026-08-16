"""Källspecifik del: hitta annonslänkar och plocka fält ur en annonssida.

VIKTIGT – läs innan du kör mot skarp sajt:
Den här filen är skriven utan tillgång till Blockets riktiga HTML (sessionen som
byggde den kom inte ut på nätet). Därför gissar den inga CSS-klasser. I stället
provas tre standardformat i tur och ordning:

  1. JSON-LD (`<script type="application/ld+json">`, schema.org Product/Offer)
  2. Next.js-data (`<script id="__NEXT_DATA__">`)
  3. Open Graph- och produktmeta (`og:title`, `product:price:amount`)

Ger ingen av dem ett pris kastas ParseError med en diagnos som säger vad sidan
faktiskt innehöll. Kör `python pipeline/scrape.py --limit 1` en gång och läs
diagnosen – då vet du vilken väg som gäller och kan snäva åt den här filen.

Byter du källa till sokbat.se: skriv om den här filen, inget annat.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterator
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

BASE_URL = "https://www.blocket.se"

# Blockets annons-URL:er ser ut som /annons/<nagot>. Samma form som i specens
# exempel (docs/PROTOTYP.md) och det enda vi behöver för att hitta annonser.
AD_PATH_RE = re.compile(r"^/annons/[\w\-/]+$")

ENGINE_BRANDS = [
    "Volvo Penta", "Yamaha", "Mercury", "Suzuki", "Honda", "Yanmar",
    "Evinrude", "Johnson", "Mariner", "Tohatsu", "Selva", "Parsun", "Nanni",
]

REGIONS = [
    "Stockholm", "Uppsala", "Södermanland", "Östergötland", "Jönköping",
    "Kronoberg", "Kalmar", "Gotland", "Blekinge", "Skåne", "Halland",
    "Västra Götaland", "Värmland", "Örebro", "Västmanland", "Dalarna",
    "Gävleborg", "Västernorrland", "Jämtland", "Västerbotten", "Norrbotten",
]

# Rimlighetsgränser. Fångar att vi råkat läsa ett organisationsnummer eller
# ett motortimtal som pris.
MIN_PRICE = 1_000
MAX_PRICE = 50_000_000


class ParseError(Exception):
    """Sidan gick inte att tolka. Meddelandet ska räcka för att fixa parsern."""


@dataclass
class ParsedAd:
    url: str
    title: str
    price: int
    year: int | None = None
    engine_brand: str | None = None
    engine_hp: int | None = None
    hours: int | None = None
    region: str | None = None
    # Vilken av de tre vägarna som gav priset. Loggas av scrape.py.
    via: str = ""


@dataclass
class Diagnosis:
    """Vad sidan innehöll när tolkningen misslyckades."""

    has_json_ld: bool = False
    has_next_data: bool = False
    meta_names: list[str] = field(default_factory=list)
    title: str = ""
    text_sample: str = ""

    def __str__(self) -> str:
        return (
            f"json-ld: {self.has_json_ld}, __NEXT_DATA__: {self.has_next_data}, "
            f"meta: {', '.join(self.meta_names[:12]) or 'inga'}\n"
            f"  <title>: {self.title[:120]}\n"
            f"  text: {self.text_sample[:200]}"
        )


def parse_int_sv(value: Any) -> int | None:
    """"1 234 567 kr" -> 1234567. Hanterar hårt mellanslag och decimaler."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).replace("\xa0", " ").replace(" ", " ")
    text = re.sub(r"[^\d,\.\s]", "", text).strip()
    if not text:
        return None
    text = re.split(r"[,\.]", text)[0]
    digits = re.sub(r"\D", "", text)
    return int(digits) if digits else None


def _walk(node: Any) -> Iterator[dict]:
    """Går igenom en godtycklig JSON-struktur och ger varje dict."""
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk(value)


def _plausible_price(value: Any) -> int | None:
    price = parse_int_sv(value)
    if price is not None and MIN_PRICE <= price <= MAX_PRICE:
        return price
    return None


# --- De tre vägarna in i sidan -------------------------------------------------


def _price_in(node: dict) -> int | None:
    """Pris på noden själv eller i dess offers – schema.org lägger priset i
    Offer medan namnet sitter på Product ovanför."""
    for key in ("price", "lowPrice", "highPrice"):
        price = _plausible_price(node.get(key))
        if price is not None:
            return price

    offers = node.get("offers")
    for offer in offers if isinstance(offers, list) else [offers]:
        if isinstance(offer, dict):
            for key in ("price", "lowPrice", "highPrice"):
                price = _plausible_price(offer.get(key))
                if price is not None:
                    return price
    return None


def _from_json_ld(soup: BeautifulSoup) -> tuple[int | None, str | None]:
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
        except (json.JSONDecodeError, TypeError):
            continue
        # _walk ger föräldern före barnen, så Product hinner före sin Offer och
        # vi får både namn och pris i samma träff.
        for node in _walk(data):
            price = _price_in(node)
            if price is not None:
                name = node.get("name") or node.get("headline")
                return price, str(name) if name else None
    return None, None


def _from_next_data(soup: BeautifulSoup) -> tuple[int | None, str | None]:
    script = soup.find("script", id="__NEXT_DATA__")
    if script is None:
        return None, None
    try:
        data = json.loads(script.string or "")
    except (json.JSONDecodeError, TypeError):
        return None, None

    for node in _walk(data):
        # Blockets annonsobjekt har historiskt haft "subject" som rubrik.
        title = node.get("subject") or node.get("title") or node.get("heading")
        if not isinstance(title, str):
            continue
        for key in ("price", "list_price", "listPrice", "amount"):
            price = _plausible_price(node.get(key))
            if price is not None:
                return price, title
            nested = node.get(key)
            if isinstance(nested, dict):
                price = _plausible_price(nested.get("value") or nested.get("amount"))
                if price is not None:
                    return price, title
    return None, None


def _from_meta(soup: BeautifulSoup) -> tuple[int | None, str | None]:
    def meta(*names: str) -> str | None:
        for name in names:
            tag = soup.find("meta", attrs={"property": name}) or soup.find(
                "meta", attrs={"name": name}
            )
            if tag and tag.get("content"):
                return tag["content"]
        return None

    price = _plausible_price(meta("product:price:amount", "og:price:amount", "price"))
    title = meta("og:title", "twitter:title")
    return price, title


# --- Fritext ------------------------------------------------------------------


def extract_year(text: str) -> int | None:
    """Årsmodell. Tar det senaste rimliga året i texten – rubriker som
    "Yamarin 79 DC 2019" har modellnumret först och året sist."""
    years = [int(m) for m in re.findall(r"\b(19[5-9]\d|20[0-4]\d)\b", text)]
    return years[-1] if years else None


def extract_engine_hp(text: str) -> int | None:
    match = re.search(r"(\d{1,4})\s*(?:hk|hp)\b", text, re.IGNORECASE)
    if not match:
        return None
    hp = int(match.group(1))
    return hp if 1 <= hp <= 2000 else None


def extract_hours(text: str) -> int | None:
    match = re.search(
        r"(\d[\d\s\xa0]{0,7})\s*(?:motortimmar|timmar|tim\b|h\b)", text, re.IGNORECASE
    )
    if not match:
        return None
    hours = parse_int_sv(match.group(1))
    return hours if hours is not None and hours <= 50_000 else None


def extract_engine_brand(text: str) -> str | None:
    lowered = text.lower()
    for brand in ENGINE_BRANDS:
        if brand.lower() in lowered:
            return brand
    return None


def extract_region(text: str) -> str | None:
    for region in REGIONS:
        if region.lower() in text.lower():
            return region
    return None


# --- Publikt API --------------------------------------------------------------


def search_url(query: str, page: int = 1) -> str:
    """Sökresultat för en modell. Sökvägen kan behöva justeras – se filens topp."""
    from urllib.parse import urlencode

    params = {"q": query, "cg": "1060"}  # 1060 = Båtar hos Blocket
    if page > 1:
        params["page"] = page
    return f"{BASE_URL}/annonser/hela_sverige?{urlencode(params)}"


def extract_ad_links(html: str, base_url: str = BASE_URL) -> list[str]:
    """Annonslänkar ur en sökresultatsida, i ordning och utan dubbletter."""
    soup = BeautifulSoup(html, "html.parser")
    links: list[str] = []
    seen: set[str] = set()

    for anchor in soup.find_all("a", href=True):
        absolute = urljoin(base_url, anchor["href"])
        parsed = urlparse(absolute)
        if not AD_PATH_RE.match(parsed.path):
            continue
        clean = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        if clean not in seen:
            seen.add(clean)
            links.append(clean)

    return links


def diagnose(html: str) -> Diagnosis:
    soup = BeautifulSoup(html, "html.parser")
    return Diagnosis(
        has_json_ld=bool(soup.find("script", type="application/ld+json")),
        has_next_data=bool(soup.find("script", id="__NEXT_DATA__")),
        meta_names=[
            m.get("property") or m.get("name")
            for m in soup.find_all("meta")
            if m.get("property") or m.get("name")
        ],
        title=soup.title.get_text(strip=True) if soup.title else "",
        text_sample=" ".join(soup.get_text(" ", strip=True).split())[:400],
    )


def parse_ad(html: str, url: str) -> ParsedAd:
    """Plockar fält ur en annonssida. Kastar ParseError om priset inte hittas."""
    soup = BeautifulSoup(html, "html.parser")

    for name, extractor in (
        ("json-ld", _from_json_ld),
        ("__NEXT_DATA__", _from_next_data),
        ("meta", _from_meta),
    ):
        price, title = extractor(soup)
        if price is None:
            continue

        if not title:
            title = soup.title.get_text(strip=True) if soup.title else ""
        body = " ".join(soup.get_text(" ", strip=True).split())
        haystack = f"{title} {body}"

        return ParsedAd(
            url=url,
            title=title.strip(),
            price=price,
            year=extract_year(title) or extract_year(body),
            engine_brand=extract_engine_brand(haystack),
            engine_hp=extract_engine_hp(haystack),
            hours=extract_hours(haystack),
            region=extract_region(haystack),
            via=name,
        )

    raise ParseError(f"hittade inget pris på {url}\n  {diagnose(html)}")
