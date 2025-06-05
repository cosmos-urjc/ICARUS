#!/usr/bin/env python3
"""
udp_viewer.py – recibe un flujo H.264 por UDP y lo muestra en una ventana.
Modifica las constantes de aquí abajo si cambias puerto, resolución, etc.
"""

import cv2
import sys

# ----------------------------------------------------------------------
#  Ajustes “hard-coded”
# ----------------------------------------------------------------------
UDP_PORT          = 5600          # Puerto donde llega el vídeo
EXPECTED_FPS      = 25            # Sólo para mostrar FPS en título (opcional)
WINDOW_TITLE      = f"UDP {UDP_PORT} – Vídeo en directo"
SHOW_FPS_OVERLAY  = True          # Pon a False si no quieres el ticker FPS
# ----------------------------------------------------------------------

# Construimos la URL FFmpeg para VideoCapture
udp_url = (
    f"udp://0.0.0.0:{UDP_PORT}"
    "?fifo_size=5000000"
    "&overrun_nonfatal=1"
)

cap = cv2.VideoCapture(udp_url, cv2.CAP_FFMPEG)
if not cap.isOpened():
    sys.exit(f"[ERROR] No se pudo abrir udp:// en el puerto {UDP_PORT}")

print(f"[INFO] Escuchando UDP H.264 en el puerto {UDP_PORT}… (q para salir)")

# Pequeño medidor de FPS
ticks, frames = cv2.getTickCount(), 0

while True:
    ok, frame = cap.read()
    if not ok:
        # Puede llegar algún paquete corrupto → intentamos continuar
        continue

    if SHOW_FPS_OVERLAY:
        frames += 1
        if frames % 30 == 0:                           # refrescamos cada 30 frames
            now = cv2.getTickCount()
            dt  = (now - ticks) / cv2.getTickFrequency()
            fps = frames / dt if dt > 0 else 0
            ticks, frames = now, 0
            cv2.setWindowTitle(WINDOW_TITLE, f"{WINDOW_TITLE} – {fps:.1f} fps")

    cv2.imshow(WINDOW_TITLE, frame)

    # Salir al pulsar 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("[INFO] Fin del programa")
