# Progetto 2 – Compressione di immagini tramite la DCT

## Descrizione

Questo progetto ha l'obiettivo di implementare e analizzare la **trasformata discreta del coseno bidimensionale (DCT2)** applicata a immagini in toni di grigio, simulando un algoritmo di compressione tipo JPEG **senza** quantizzazione. Il progetto si articola in due parti:

1. **Implementazione e confronto della DCT2 personalizzata** con la versione fornita dalla libreria del linguaggio scelto.
2. **Sviluppo di un software** che esegue compressione e decompressione di immagini basata sulla DCT2, con possibilità di impostare parametri di blocco e soglia di taglio delle frequenze.

## Requisiti

* Ambiente **open-source** (es. Python, Julia, C++, ecc. – MATLAB non ammesso)
* Libreria che supporti la DCT e IDCT bidimensionale (es. `scipy.fftpack`, `cv2`, `numpy.fft`, ecc.)
* Immagini `.bmp` in toni di grigio
* Interfaccia per la selezione dell’immagine e dei parametri

## Parte 1 – Confronto delle prestazioni della DCT2

* Implementare la DCT2 "fatta in casa", eseguendo DCT1 su righe e poi colonne
* Usare anche la DCT2 della libreria (veloce, tipo FFT)
* Eseguire test su array quadrati N×N (N crescente) e confrontare i tempi su grafico semilogaritmico
* Aspettativa:

  * DCT fatta in casa: complessità \~ O(N³)
  * DCT veloce: \~ O(N² log N)
* Includere nella relazione un'analisi dei risultati ottenuti

## Parte 2 – Compressione di immagini

### Funzionalità richieste

* Caricamento di un’immagine `.bmp` in toni di grigio da interfaccia
* Scelta dei parametri:

  * `F`: dimensione dei blocchi (macro-blocchi F×F)
  * `d`: soglia di taglio frequenze, 0 ≤ d ≤ 2F − 2
* Elaborazione:

  * Suddivisione dell’immagine in blocchi F×F (scartando eventuali margini)
  * Per ogni blocco:

    * Applicazione della DCT2
    * Annullamento delle componenti ckℓ con k + ℓ ≥ d
    * Ricostruzione tramite IDCT2
    * Clipping dei valori fuori da \[0, 255]
  * Ricostruzione dell'immagine compressa
* Visualizzazione dell'immagine originale e compressa affiancate

### Note tecniche

* Verificare la correttezza della DCT implementata confrontando output su blocchi noti
* Documentare:

  * Linguaggio usato
  * Libreria per DCT/IDCT
  * Struttura del codice
  * Esperimenti con immagini (anche quelle fornite nel sito e-learning)
  * Analisi qualitativa dei risultati ottenuti
