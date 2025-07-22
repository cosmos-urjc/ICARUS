import cv2
print(cv2.getBuildInformation())

# --- Configuración del Pipeline de GStreamer ---
# Puerto para el video primario de OpenHD
PORT = 5600

# Pipeline para H.264. Es el formato más común en OpenHD.
# udpsrc: Recibe los paquetes UDP en el puerto especificado.
# caps: Especifica el formato del video (RTP con H.264).
# rtph264depay: Extrae los datos H.264 del paquete RTP.
# avdec_h264: Decodifica el stream H.264.
# videoconvert: Convierte el formato de color a uno compatible con OpenCV.
# appsink: Permite que OpenCV capture los frames.

pipeline_h264 = (
    f"udpsrc port={PORT} "
    f"! application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264 "
    f"! rtph264depay ! avdec_h264 ! videoconvert ! appsink sync=false"
)

# --- Captura de Video ---
# Cambia a pipeline_h265 si tu stream usa ese códec.
cap = cv2.VideoCapture(pipeline_h264, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("❌ Error: No se pudo abrir el stream de video.")
    print("Verifica que GStreamer esté instalado correctamente y que el stream de OpenHD esté activo.")
    exit()

print("✅ Stream de video abierto correctamente. Mostrando ventana...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Esperando stream...")
        continue

    # Muestra el frame en una ventana
    cv2.imshow('OpenHD Video Stream', frame)

    # Presiona 'q' para salir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Liberar recursos ---
cap.release()
cv2.destroyAllWindows()
print("🛑 Programa finalizado y recursos liberados.")