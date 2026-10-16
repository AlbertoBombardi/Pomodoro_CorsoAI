# Pomodoro anti-telefono
Progetto finale della Scuola "Python, Deep Learning and Computer Vision".
Un timer Pomodoro (25 min studio / 5 min pausa) che usa la webcam per
accorgersi se, invece di studiare, stai usando il telefono.

## Come funziona
1. Ogni 10 secondi la webcam si accende, scatta un'immagine e si rispegne
   (per contenere il consumo di batteria).
2. **MediaPipe** (BlazeFace) verifica se c'è un volto davanti allo schermo.
3. **YOLO11s** (pre-addestrato su COCO, nessun training) cerca le classi
   *persona* e *telefono*. Il telefono conta solo se almeno il 50% del suo
   riquadro cade dentro quello di una persona, per ignorare un telefono
   appoggiato sulla scrivania.
4. Ogni rilevamento costa una "vita": dopo 3 avvisi vocali sempre più severi,
   alla quarta volta la sessione viene interrotta.
5. Se non vengono rilevati né volto né telefono, il timer si sospende e
   riparte al tuo ritorno.
6. Alla chiusura (Ctrl+C) viene mostrato un riepilogo: sessioni iniziate,
   durata totale e tempo stimato passato al telefono.

## Requisiti
- macOS (gli avvisi vocali usano il comando `say`)
- Python con `opencv-python`, `ultralytics`, `mediapipe`
- File del modello `blaze_face_short_range.tflite` nella cartella dello script
  (i pesi `yolo11s.pt` vengono scaricati automaticamente da Ultralytics)

## Avvio
    python pomodoro3.py

## Limiti noti
- Il controllo è a campione (un frame ogni 10 s), quindi un uso breve del
  telefono tra due controlli può sfuggire.
- Con soglia di confidenza bassa (0.2) sono possibili falsi positivi.
- Funziona solo su macOS per via della sintesi vocale.
