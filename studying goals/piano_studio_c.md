# Piano operativo di roccix: dal C all'inferenza Llama

## Assunzione e ruolo

Si assume soltanto che tu conosca C; nessuna conoscenza AI e data per scontata. Segui le unita del piano condiviso senza saltarne nessuna. Il tuo ruolo iniziale e trasformare ogni operazione gia capita a mano in loop, layout e memoria espliciti; non e ottimizzare.

Prima di ogni funzione compila:

```text
scopo; input/output con shape e dtype; formula/pseudocodice;
elementi e byte; buffer letti/modificati; ownership e lifetime;
caso manuale; expected NumPy; atol/rtol.
```

Mantieni una versione semplice anche quando aggiungerai quella veloce. In debug usa warning completi e sanitizer.

## Lavoro concreto per le unita condivise

### 0-1. Sistema completo e tokenizer

1. Compila ed esegui il riferimento seguendo il README.
2. Traccia `main -> generate -> forward -> sample`, senza studiare ancora il corpo matematico.
3. Disegna file, pesi persistenti, attivazioni, cache e output.
4. Implementa un vocabolario giocattolo, prima decode ID->stringa, poi encode.
5. Aggiungi i merge BPE definiti con JD.
6. Testa ID fuori range, stringa vuota, spazi, punteggiatura e UTF-8 come byte.
7. Per ogni allocazione annota proprietario e cleanup.

Prompt LLM: “Mostrami dieci variabili tipiche di un LLM e fammele classificare come configurazione, peso, attivazione o cache”.

### 2. Layout e ponte binario

1. Alloca e stampa array 1D, 2D e 3D linearizzati.
2. Scrivi funzioni di offset per `[row,col]` e `[layer,pos,head,component]`.
3. Predici cinque indirizzi relativi prima di eseguire.
4. Leggi il fixture `float32` di JD validando magic, versione, shape, dtype, endian e lunghezza.
5. Produci lo stesso formato da C per Python.
6. Gestisci `fopen`, letture corte, dimensioni invalide, moltiplicazioni `size_t` in overflow e allocazioni fallite.
7. Usa tipi a larghezza fissa nel formato.

### 3-4. Primitive e precisione

Implementa una funzione e un test alla volta: vector add, element-wise multiply, scalar multiply, sum, max, argmax, dot, matvec row-major, mean-square, RMS, confronto `atol/rtol`.

Per ciascuna:

1. ricevi shape e expected values da JD;
2. prova zero, negativi, frazioni, valori grandi e dimensione zero se ammessa;
3. stampa il primo indice divergente;
4. spiega `w[i*d+j]` con un disegno della memoria;
5. sperimenta ordini di accumulazione e overflow dei tipi interi.

Prompt: “Dammi casi di test difficili per matvec `float32`, senza implementazione; indica il bug rilevato da ciascuno”.

### 5-8. Rete minima e primitive Llama

1. Scrivi `linear(out,W,x,rows,cols)` usando matvec.
2. Costruisci il forward `3->2->2`; nessuna allocazione dentro la primitiva.
3. Confronta ogni layer, non solo l'output.
4. Implementa softmax ingenua, provoca overflow, poi rendila stabile.
5. Implementa greedy; RNG/sampling, temperature e top-p soltanto dopo gli expected di JD.
6. Implementa embedding lookup come selezione di riga con validazione ID.
7. Scomponi RMSNorm in mean-square, epsilon, inverse RMS, scale element-wise.
8. Implementa sigmoid, SiLU, gating SwiGLU e residual; documenta l'aliasing consentito.
9. Solo dopo confronta ciascuna funzione con `run.c`.

### 9-10. Attention minima

1. Crea un programma autonomo: una head, due componenti, prima uno poi due token.
2. Usa pesi/expected preparati con JD.
3. Stampa q, k, v, raw score, scaled score, probabilita e output.
4. Rendi visibile ogni cella esclusa dalla causal mask.
5. Estendi a tre token.
6. Confronta ogni valore prima di aprire il loop equivalente in `forward()`.

Prompt: “Dammi un indice in un tensore flat `[position,head,component]` e attendi che calcoli offset e byte”.

### 11-13. Multi-head, GQA, RoPE e cache

1. Fissa e documenta il layout prima di scrivere i loop.
2. Implementa due heads separate, concatenazione e `Wo`.
3. Aggiungi la mappa esplicita query-head->KV-head; usa valori sentinel per trovare head errate.
4. Implementa una rotazione di due float; testa 0°, 90° e norma.
5. Estendi RoPE a coppie, frequenze e posizioni; controlla i limiti q/KV.
6. Definisci la KV cache con shape e byte checked.
7. Per tre posizioni stampa base, offset, indirizzo e contenuto scritto.
8. Inizializza il futuro con sentinel e prova che non venga letto.
9. Confronta cached contro ricalcolo completo.
10. Documenta create/reset/reuse/destroy.

### 14. Blocco Llama

1. Funzioni separate per attention e feed-forward.
2. Prealloca `RunState`; nessun `malloc` nel forward.
3. Tabella di tutti i buffer e di quelli riutilizzabili.
4. Trace attivabile per layer/posizione scelti.
5. Prima blocco artificiale, poi layer 0/position 0 reale.
6. Fermati alla prima divergenza; non allargare la tolleranza senza causa.

### 15. Checkpoint

1. Tabella campo/offset/byte/significato della configurazione.
2. Deriva, non copiare, shape e byte di ogni peso.
3. Usa add/multiply checked per `size_t`.
4. Testa file corto, header assurdo e allocazione fallita.
5. Mappa ogni puntatore a un intervallo del file.
6. Fai confrontare a JD campioni e checksum.

### 16. Motore float32

Ordine rigido: embedding; un layer/posizione; tutti i layer/una posizione; norm finale; logits; argmax; 2 token; 10; 50; tokenizer/decode; 200 greedy; sampling opzionale.

Build minime:

```text
debug:    -O0 -g -Wall -Wextra -Wpedantic
sanitize: -O1 -g -fsanitize=address,undefined
release:  -O3 -DNDEBUG
```

Per ogni mismatch segui l'ordine del protocollo condiviso. La build sanitizer deve essere pulita.

### 17-19. Quantizzazione/Q8_0

1. Quantize/dequantize di un gruppo; gestisci zero, round e clip.
2. Misura errore e storage.
3. Dimostra il bound e usa accumulatore `int32`.
4. Leggi `export.py`/`runq.c` solo dopo l'esercizio indipendente.
5. Loader versionato: valida type, group size, shape e file length.
6. Confronta i byte di fixture zero/saturazione/outlier/multigruppo.
7. Implementa activation quantization, dot e matvec.
8. Sostituisci una matvec per volta.
9. Tieni distinti bug C-vs-NumPy Q8_0 ed errore Q8_0-vs-float.

### 20. Ottimizzazione

1. Congela baseline e comando.
2. Profila prima di scegliere.
3. Elimina allocazioni per token; misura.
4. Migliora localita/ordine loop; misura.
5. OpenMP soltanto su righe indipendenti; misura diversi thread.
6. Leggi il report di vectorization.
7. SIMD esplicito soltanto sul collo misurato.
8. Dopo ogni cambio: unit test, cross-test, sanitizer e stesso benchmark.

## Verifica personale finale

Devi saper spiegare ogni operazione senza gergo, ricostruire attention minima a mano, prevedere shape/offset/byte/ownership/lifetime, localizzare la prima divergenza, produrre 200 token greedy uguali, spiegare Q8_0 e dimostrare ogni speedup. Sapere C non autorizza a saltare il significato matematico dei dati.
