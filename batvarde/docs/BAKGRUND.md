# Bakgrund: varför vi bygger det här

## Idén i en mening
En gratis, oberoende tjänst som visar vad begagnade båtar faktiskt säljs för i Norden – så att
privatpersoner kan värdera, köpa och sälja rätt, medan mäklare, banker och försäkringsbolag
betalar för leads och prisdata bakom. Arbetsnamn: Båtvärde. Förebild: Bilpriser (KVD) för bilar.

## Problemet
- Sverige har ~1,5 miljoner fritidsbåtar och ~800 000 båtägande hushåll (Transportstyrelsen 2025).
- Blocket har ~17 000 båtannonser i säsong och är de facto prisreferens – men visar bara
  annonspriser (listpriser). Annonspris och slutpris skiljer ofta 10–30 %.
- Det finns inget båtregister sedan 1992 och ingen försäljningsstatistik (Sweboat/SCB kan inte
  leverera). Ingen aktör i Norden säljer sålda-priser-data eller värderings-API.
- Alla gratis "båtvärderingar" (VärderaMinBåt.se, Båtvärdering.se, Yachtsale, Båtagent, Navark)
  är mäklarnas leadgen. SäljaDinBåt.se (lanserad feb 2026) jämför mäklarofferter men bygger sitt
  riktpris på listpriser och tar bara båtar över 200 000 kr, bara säljare.

## Vår skillnad
Vi äger sanningen om priset: sålda/borttagna annonser, auktionsutfall, prishistorik, tid på
marknaden, prissänkningar. Vi tjänar både köpare och säljare, alla prisklasser. Mäklarleads är
ett av flera intäktsben, inte hela affären.

## Kunder och intäkter (senare, inte i prototypen)
- Privat säljare/köpare: gratis. Ger trafik, data och leads.
- Båtmäklare (~150 i SE): betalar per lead eller andel av arvode (5–10 % av slutpris).
- Finansbolag (Wasa Kredit, Nordea, Swedbank Finans, Santander): värderings-API för båtlån.
- Försäkringsbolag (Svedea, Atlantica, If, Länsförsäkringar): marknadsvärde vid skada/stöld.
- Affiliate: båtlån och båtförsäkring.

## Team
Elliot (business/data, sälj, finansnätverk) och Matija (produkt/teknik, community).
Vi har tidigare byggt pokeSwe: scraping, prisnormalisering, SEO, drop/restock-bevakning.
Samma motor, ny marknad.

## Vad prototypen ska bevisa (och bara det)
1. Vi kan samla båtannonser automatiskt och normalisera per modell/årgång.
2. Vi ser när annonser försvinner (proxy för sålt) och sista pris.
3. Modellsida med prisintervall aktiva vs troligen sålda, prishistorik, dagar på marknaden.
4. "Är den här annonsen rätt prissatt?" – det ingen annan gör.

Allt annat (login, leads, affiliate, API, bevakning, PDF, Norge, deploy) är medvetet utanför.
