import Link from 'next/link'
import { WINDOW_DAYS } from '@/lib/stats'
import { CONDITION_STEP, DEVIATION_THRESHOLD, YEAR_SPAN } from '@/lib/valuation'

export const metadata = { title: 'Så räknar vi – Båtvärde' }

export default function SaRaknarViPage() {
  return (
    <div className="flex max-w-2xl flex-col gap-5">
      <h1 className="text-2xl font-semibold">Så räknar vi</h1>

      <p>
        <span className="font-medium">Var siffrorna kommer ifrån.</span> Vi hämtar båtannonser
        flera gånger i veckan och normaliserar dem per modell och årsmodell, så att
        &quot;Yamarin 79&quot;, &quot;79 DC&quot; och &quot;79dc&quot; hamnar på samma modell. För
        varje annons sparar vi pris, år, motor, region och när vi såg den första och senaste
        gången. Ändras priset sparar vi både det gamla och det nya, så att prissänkningar går att
        räkna.
      </p>

      <p>
        <span className="font-medium">Vad &quot;troligen sålt&quot; betyder.</span> Det finns
        inget båtregister och ingen offentlig försäljningsstatistik i Sverige, så ingen kan visa
        bekräftade slutpriser. Det vi kan se är när en annons försvinner. En annons som funnits i
        minst två hämtningar och sedan saknats i två räknar vi som borttagen, och priset den hade
        sist blir vårt &quot;troligen sålt&quot;-pris. Det är en uppskattning, inte ett kvitto:
        en del annonser tas bort utan att båten sålts. Men skillnaden mot listpriserna är den
        närmaste sanning som finns att få, och den är alltid större än noll.
      </p>

      <p>
        <span className="font-medium">Hur intervallen räknas.</span> Alla siffror bygger på
        annonser från de senaste {WINDOW_DAYS} dagarna. Vi visar medianen och spannet mellan
        fjärde och sjätte tiondelen av priserna (p25–p75) – inte snittet, som en enda dyr båt
        annars drar iväg. Konfidensen säger hur mycket underlag vi har: hög från 30
        observationer, medel från 10, låg därunder. I värderingen jämför vi med samma modell inom
        ±{YEAR_SPAN} årsmodeller, låter borttagna annonser väga dubbelt så tungt som aktiva, och
        justerar {Math.round(CONDITION_STEP * 100)} % per steg du flyttar skicket från
        &quot;normalt för åldern&quot;. I annonskollen är gränsen ±{DEVIATION_THRESHOLD} % från
        medianen innan vi kallar ett pris högt eller lågt.
      </p>

      <p className="rounded-lg border border-border bg-card p-4 text-sm text-muted-foreground">
        Det här är en prototyp. Underlaget är litet, täcker tio modeller och ska inte användas
        som beslutsunderlag vid en affär.{' '}
        <Link className="underline" href="/">
          Tillbaka till modellerna
        </Link>
        .
      </p>
    </div>
  )
}
