import time
import math
import subprocess                  # per lanciare comandi del Mac, come la sintesi vocale "say"
import cv2
from ultralytics import YOLO
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# --- IMPOSTAZIONI ---
# per fermarlo ctrl + c

DURATA_STUDIO = 25 * 60            # secondi di studio per sessione
DURATA_PAUSA = 5 * 60              # secondi di pausa
INTERVALLO = 15                    # secondi tra un controllo e l'altro
VITE = 3                           # beccate "perdonate": alla successiva la sessione si ferma
SOGLIA_CONFIDENZA = 0.3
FRAME_DI_RISCALDAMENTO = 8
VOCE = "Samantha"                  # voce inglese (USA) del Mac
CLASSE_PERSONA = 0
CLASSE_TELEFONO = 67
SOVRAPPOSIZIONE_MINIMA = 0.5       # frazione del riquadro del telefono che deve stare dentro quello della persona
SOGLIA_VOLTO = 0.3                 # confidenza minima di MediaPipe per dire "c'è un volto"
# Se True, anche quando il volto non si vede (es. testa china sul telefono) cerco comunque il telefono
# prima di sospendere il timer. Se False, senza volto il timer si ferma e basta.
CONTROLLA_TELEFONO_SENZA_VOLTO = True

MESSAGGI = [
    "Come on, {minuti} until the break!",
    "Stop it, {minuti} until the break!",
    "Put the phone down, {minuti} until the break.",
]

# --- RILEVAZIONE PERSONA & TELEFONO (YOLO) ---
rilevatore_yolo = YOLO("yolo11s.pt")

# --- RILEVATORE DI VOLTI (MediaPipe) ---
rilevatore_volti = vision.FaceDetector.create_from_options(
    vision.FaceDetectorOptions(
        base_options=mp_python.BaseOptions(model_asset_path="blaze_face_short_range.tflite"),
        min_detection_confidence=SOGLIA_VOLTO,
    )
)


def parla(testo):
    """Scrive il testo nel terminale e lo pronuncia con la voce del Mac."""
    print(testo)
    # Popen avvia la voce "in parallelo": il programma non resta bloccato mentre parla
    subprocess.Popen(["say", "-v", VOCE, testo])


def testo_minuti(fine):
    """Trasforma il tempo che manca a 'fine' in un testo tipo '12 minutes left'."""
    minuti = math.ceil((fine - time.time()) / 60)   # arrotondo per eccesso
    if minuti <= 1:
        return "1 minute left"
    return f"{minuti} minutes left"


def scatta_foto():
    """Accende la webcam, prende un'immagine e la rispegne."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return None
    for _ in range(FRAME_DI_RISCALDAMENTO):
        cap.read()
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def vedo_volto(frame):
    """True se MediaPipe trova almeno un volto nell'immagine."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)     # OpenCV usa BGR, MediaPipe vuole RGB
    immagine = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    risultato = rilevatore_volti.detect(immagine)
    return len(risultato.detections) > 0


def sovrapposizione(telefono, persona):
    """Che frazione del riquadro del telefono cade dentro il riquadro della persona (da 0 a 1).
    Ogni riquadro è [x1, y1, x2, y2]: angolo in alto a sinistra e angolo in basso a destra."""
    # Il rettangolo comune ai due riquadri (l'intersezione)
    x1 = max(telefono[0], persona[0])
    y1 = max(telefono[1], persona[1])
    x2 = min(telefono[2], persona[2])
    y2 = min(telefono[3], persona[3])
    area_comune = max(0, x2 - x1) * max(0, y2 - y1)   # 0 se non si toccano
    area_telefono = (telefono[2] - telefono[0]) * (telefono[3] - telefono[1])
    return area_comune / area_telefono if area_telefono > 0 else 0


def analizza_foto(frame):
    """Restituisce (c'è una persona?, c'è un telefono sovrapposto alla persona?)."""
    results = rilevatore_yolo(frame, classes=[CLASSE_PERSONA, CLASSE_TELEFONO],
                    conf=SOGLIA_CONFIDENZA, verbose=False)
    boxes = results[0].boxes

    # Separo i riquadri delle persone da quelli dei telefoni
    persone = [b.xyxy[0].tolist() for b in boxes if int(b.cls) == CLASSE_PERSONA]
    telefoni = [b.xyxy[0].tolist() for b in boxes if int(b.cls) == CLASSE_TELEFONO]

    # Conta solo un telefono che sta (almeno in parte) dentro il riquadro di una persona
    telefono_su_persona = any(sovrapposizione(t, p) >= SOVRAPPOSIZIONE_MINIMA
                              for t in telefoni for p in persone)

    return len(persone) > 0, telefono_su_persona


def sessione_studio(numero):
    """Una sessione pomodoro. Restituisce (completata?, secondi passati al telefono)."""
    parla(f"Starting session {numero}. Happy studying!")
    fine = time.time() + DURATA_STUDIO
    beccate = 0
    secondi_al_telefono = 0
    assente = False                # True quando il timer è sospeso perché non ci sei
    ultimo_controllo = time.time()

    while time.time() < fine:
        # Se al controllo precedente non c'eri, il tempo passato non conta: sposto in avanti la fine
        adesso = time.time()
        if assente:
            fine += adesso - ultimo_controllo
        ultimo_controllo = adesso

        frame = scatta_foto()
        if frame is None:
            print("Can't read from the webcam, retrying at the next check.")
        else:
            # PASSO 1: cerco un volto con MediaPipe
            volto = vedo_volto(frame)

            # PASSO 2: cerco persona e telefono con YOLO (se c'è il volto, o se ho scelto
            # di controllare il telefono anche senza volto)
            if volto or CONTROLLA_TELEFONO_SENZA_VOLTO:
                persona, telefono = analizza_foto(frame)
            else:
                persona, telefono = False, False

            # Niente volto e niente telefono in mano: non sei alla scrivania, sospendo il timer
            if not volto and not telefono:
                if not assente:
                    print("No face in front of the webcam: timer paused.")
                assente = True
                print(f"[{time.strftime('%H:%M:%S')}] {'TIMER PAUSED':<24} ({testo_minuti(fine)})")
                time.sleep(INTERVALLO)
                continue

            if assente:
                assente = False
                parla("Get moving, Alberto")

            in_uso = persona and telefono

            # Ogni controllo con il telefono in uso è una beccata e costa una vita
            if in_uso:
                secondi_al_telefono += INTERVALLO
                beccate += 1
                if beccate > VITE:
                    return False, secondi_al_telefono   # vite finite: sessione interrotta
                parla(MESSAGGI[beccate - 1].format(minuti=testo_minuti(fine)))

            ora = time.strftime("%H:%M:%S")
            stato = ">>> PHONE IN USE <<<" if in_uso else "studying"
            vite_rimaste = VITE - beccate
            print(f"[{ora}] {stato:<24} lives: {'❤️ ' * vite_rimaste}  ({testo_minuti(fine)})")

        time.sleep(INTERVALLO)

    return True, secondi_al_telefono


def pronto_per_ripartire():
    """Chiede (a voce e nel terminale) se iniziare un'altra sessione."""
    parla("Are you ready for another session?")
    risposta = input("Are you ready for another session? (y/n): ").strip().lower()
    return risposta in ("y", "yes")


# --- PROGRAMMA PRINCIPALE ---
numero_sessione = 1
totale_telefono = 0
inizio = time.time()

try:
    while True:
        completata, secondi = sessione_studio(numero_sessione)
        totale_telefono += secondi

        if completata:
            parla("Great job, session complete! Enjoy a five minute break.")
        else:
            print("\n" + "=" * 55)
            print("   YOU'RE NOT FOCUSED: REST FOR 5 MINUTES, THEN START AGAIN")
            print("=" * 55 + "\n")
            parla("You're not focused. Rest for five minutes, then start again.")

        fine_pausa = time.strftime("%H:%M", time.localtime(time.time() + DURATA_PAUSA))
        print(f"Break until {fine_pausa}...")
        time.sleep(DURATA_PAUSA)

        if not pronto_per_ripartire():
            break
        numero_sessione += 1

except KeyboardInterrupt:
    pass                           # Ctrl+C: esco dal ciclo e vado al riepilogo

durata = time.time() - inizio
print("\n--- SUMMARY ---")
print(f"Sessions started:   {numero_sessione}")
print(f"Total time:         {int(durata // 60)} min")
print(f"Time on phone:      {totale_telefono // 60} min")
parla("See you next time!")