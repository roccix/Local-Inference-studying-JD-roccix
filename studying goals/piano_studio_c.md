# Percorso pratico per chi conosce C

## Obiettivo

Il tuo compito non e solo riscrivere `run.c`. Alla fine del percorso dovrai saper spiegare, implementare e misurare un motore di inferenza Llama 2 in C, prima in `float32` e poi quantizzato in `int8`.

Il risultato finale dovra:

1. caricare un checkpoint;
2. convertire il prompt in token;
3. eseguire il Transformer;
4. generare gli stessi token del riferimento Python in modalita greedy;
5. caricare pesi Q8_0 ed eseguire matmul quantizzate;
6. confrontare correttezza, memoria e velocita.

Non devi studiare tutta la matematica in anticipo. Per ogni operazione segui questo ciclo:

```text
esempio numerico a mano
-> versione Python leggibile
-> versione C
-> confronto automatico
-> uso nel modello
```

## Regole di lavoro

- Scrivi sempre la forma dei dati: per esempio `W[4][3] * x[3] -> y[4]`.
- Chiedi al tuo compagno di spiegare la formula con numeri piccoli, non soltanto con simboli.
- Spiega al tuo compagno puntatori, layout, lifetime e ownership della memoria.
- Non ottimizzare una funzione senza un test numerico.
- Usa inizialmente generazione greedy (`-t 0.0`), per avere risultati riproducibili.
- Mantieni una versione semplice di riferimento anche quando aggiungerete ottimizzazioni.

## Fase 1 — Orientamento nel repository

Compila ed esegui i test locali:

```bash
make rundebug
make testcc
```

Con i file del modello 260K disponibili:

```bash
./run test/stories260K.bin -z test/tok512.bin -t 0.0 -n 10
```

Leggi `run.c` in questo ordine:

1. `main`;
2. `Config`, `TransformerWeights`, `RunState`, `Transformer`;
3. `build_transformer` e caricamento del checkpoint;
4. `malloc_run_state` e funzioni di cleanup;
5. `rmsnorm`, `softmax`, `matmul`;
6. `forward`, un blocco concettuale alla volta;
7. `sample_argmax` e `sample`;
8. tokenizer;
9. `generate`.

Per ogni funzione annota:

```text
Scopo:
Input e dimensioni:
Output e dimensioni:
Memoria letta:
Memoria modificata:
Chi possiede la memoria:
Test disponibile:
```

Rimanda `runq.c`, `train.py` ed `export.py` finche l'inferenza `float32` non sara chiara.

## Fase 2 — Matematica minima attraverso C

Implementa queste operazioni in un piccolo file separato:

1. somma e prodotto elemento per elemento;
2. prodotto scalare;
3. prodotto matrice-vettore;
4. media quadratica;
5. RMSNorm;
6. softmax stabile;
7. SiLU;
8. argmax.

Per ogni operazione:

- calcola un caso con 2–4 elementi a mano;
- ricevi dal tuo compagno il risultato Python/NumPy;
- implementa la funzione C con loop espliciti;
- confronta con tolleranza, non con uguaglianza esatta;
- prova anche input nulli, negativi e di grande valore.

Concetti matematici da saper spiegare in linguaggio semplice:

- un vettore e una lista ordinata di numeri;
- una matrice trasforma un vettore in un altro vettore;
- il prodotto scalare misura quanto due vettori puntano nella stessa direzione;
- softmax trasforma punteggi in probabilita;
- RMSNorm controlla la scala di un vettore;
- un logit e un punteggio non ancora normalizzato.

## Fase 3 — Forward pass Llama

Studia `forward()` in questo ordine:

1. lookup dell'embedding del token;
2. loop sui layer;
3. RMSNorm prima dell'attenzione;
4. proiezioni Q, K, V;
5. RoPE come rotazione di coppie di valori;
6. scrittura e lettura della KV cache;
7. score di attenzione e softmax;
8. somma pesata dei value;
9. proiezione di output e connessione residuale;
10. RMSNorm e blocco SwiGLU;
11. normalizzazione finale e logits.

Prima di passare oltre, devi poter disegnare questo flusso e indicare la dimensione di ogni buffer.

### Esercizio obbligatorio: attenzione minuscola

Implementa un'attenzione con:

```text
sequence_length = 2
head_size = 2
n_heads = 1
```

Stampa `q`, `k`, `v`, score, probabilita e output. Il tuo compagno produrra gli stessi valori con NumPy. I due risultati devono coincidere entro una tolleranza concordata.

### Esercizio obbligatorio: KV cache

Esegui tre posizioni. Stampa l'indirizzo e il contenuto della porzione di cache usata a ogni passo. Devi saper spiegare perche K e V precedenti non vengono ricalcolati.

## Fase 4 — Il vostro motore `float32`

Non copiare la struttura monolitica di `run.c`. Una possibile organizzazione e:

```text
include/
  checkpoint.h
  ops.h
  transformer.h
  tokenizer.h
  sampler.h
src/
  checkpoint.c
  ops.c
  transformer.c
  tokenizer.c
  sampler.c
  main.c
tests/
```

Ordine di implementazione:

1. primitive numeriche;
2. formato e caricamento del checkpoint;
3. embedding;
4. RoPE;
5. attenzione e KV cache;
6. SwiGLU;
7. Transformer block;
8. forward completo;
9. greedy sampling;
10. tokenizer e generazione;
11. temperature e top-p.

Usa almeno tre build:

```text
debug:    -O0 -g -Wall -Wextra -Wpedantic
sanitize: -O1 -g -fsanitize=address,undefined
release:  -O3 -DNDEBUG
```

Il tuo compito principale durante questa fase e rendere espliciti:

- ownership delle allocazioni;
- dimensione in byte di ogni tensore;
- offset dei tensori nel checkpoint;
- layout row-major delle matrici;
- lifetime di pesi, attivazioni e cache;
- controllo di overflow nei calcoli delle dimensioni.

### Criterio di completamento

Con il modello 260K, il vostro C deve produrre gli stessi token greedy del riferimento Python per almeno 200 token. Alcuni tensori intermedi scelti devono coincidere entro la tolleranza documentata. La build con sanitizer non deve riportare errori.

## Fase 5 — Quantizzazione Q8_0

Studia prima un singolo vettore. Per quantizzazione simmetrica `int8`:

```text
scale = max(abs(x)) / 127
q[i] = round(x[i] / scale)
x_ricostruito[i] = q[i] * scale
```

Gestisci esplicitamente il caso in cui tutti i valori siano zero.

Implementa e testa:

1. `quantize_vector`;
2. `dequantize_vector`;
3. errore massimo e medio;
4. quantizzazione per tensore;
5. quantizzazione per riga;
6. quantizzazione a gruppi di 64, 32, 16 e 8 valori.

Inserisci un outlier nel vettore e osserva come cambia l'errore. Questo rende concreto il motivo dei gruppi.

Poi leggi il codice reale in questo ordine:

1. `quantize_q80` in `export.py`;
2. export versione 2 in `export.py`;
3. `QuantizedTensor` in `runq.c`;
4. `quantize` e `dequantize`;
5. `matmul` quantizzata;
6. forward quantizzato.

Per un gruppo, il dot product usa l'idea:

```text
dot(float_x, float_w)
~= scale_x * scale_w * sum(int_x[i] * int_w[i])
```

L'accumulatore deve essere piu largo degli operandi `int8`. Calcola un limite superiore prima di scegliere il tipo.

### Criterio di completamento

Il motore quantizzato deve:

- caricare un formato versionato;
- verificare quantization type e group size;
- concordare con il riferimento quantizzato Python;
- misurare la deviazione dal modello `float32`;
- riportare dimensione del file, RAM e token al secondo.

## Fase 6 — Ottimizzazione

Ottimizza soltanto dopo avere salvato una baseline. Procedi cosi:

1. elimina allocazioni nel loop per token;
2. misura i tempi per operazione;
3. migliora la localita degli accessi;
4. parallelizza le righe indipendenti della matmul con OpenMP;
5. controlla i report di auto-vectorization del compilatore;
6. valuta SIMD esplicito soltanto sul vero collo di bottiglia.

Ogni benchmark deve registrare:

```text
CPU, compilatore, flag, thread, modello, context length,
numero di token, dimensione file, RAM, token/s e metrica di errore
```

## Come insegnare e come imparare dal compagno

Quando insegni C, non limitarti alla sintassi. Fai prevedere al tuo compagno:

- quanti byte vengono allocati;
- dove punta ogni puntatore;
- quando un buffer diventa invalido;
- come una matrice 2D viene linearizzata;
- quali loop possono essere eseguiti in parallelo.

Quando studi matematica, chiedi sempre:

1. Quali numeri entrano?
2. Quali numeri escono?
3. Quali sono le dimensioni?
4. Posso calcolare un caso piccolo a mano?
5. Dove vengono memorizzati questi numeri in C?

Il tuo traguardo non e conoscere ogni formula a memoria. E saper trasformare una formula in loop corretti, testati e misurabili.

