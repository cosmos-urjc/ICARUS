#!/usr/bin/env python3
"""
Detección de UAV en vídeos utilizando Ultralytics YOLOv8 en GPU NVIDIA.
Procesa todos los vídeos en /videos y guarda los resultados con bounding boxes verdes y probabilidades en /video/mirados.
Requisitos:
    pip install ultralytics opencv-python
    Tener un modelo entrenado (best.pt) o usar yolov8n.pt sin entrenamiento.
    GPU NVIDIA con cuda disponible.
"""
import cv2
from ultralytics import YOLO
from pathlib import Path
import torch
import shutil

def process_video(video_path: Path, output_path: Path, model: YOLO, conf_threshold: float, device: str):
    # Verifica si el archivo de video existe. Si no existe, imprime un mensaje y termina la función.
    if not video_path.exists():
        print(f"File not found: {video_path}")
        return
    
    # Abre el archivo de video utilizando OpenCV.
    cap = cv2.VideoCapture(str(video_path))
    # Verifica si el archivo de video se pudo abrir correctamente. Si no, imprime un mensaje y termina la función.
    if not cap.isOpened():
        print(f"Error: Unable to open file: {video_path}")
        return

    # Obtiene la tasa de fotogramas (FPS) del video. Si no se puede obtener, usa un valor predeterminado de 30.0.
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    # Obtiene el ancho y alto del video en píxeles.
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Verifica si el directorio de salida existe. Si no, lo crea.
    if not output_path.parent.exists():
        print(f"Creating directory: {output_path.parent}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configura el códec de video para guardar el archivo procesado en formato MP4.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # Crea un objeto VideoWriter para escribir el video procesado.
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    # Verifica si el objeto VideoWriter se pudo abrir correctamente. Si no, libera el recurso del video y termina la función.
    if not writer.isOpened():
        print(f"Error: Unable to open writer for: {output_path}")
        cap.release()
        return

    # Inicializa variables para acumular estadísticas:
    # - sum_conf: suma de las probabilidades de detección.
    # - total_detections: número total de detecciones realizadas.
    # - detected_frame_count: número de fotogramas que contienen UAVs detectados.
    sum_conf = 0.0
    total_detections = 0
    detected_frame_count = 0

    # Desactiva el cálculo de gradientes para acelerar el procesamiento (modo de inferencia).ºª
    with torch.no_grad():
        # Itera sobre cada fotograma del video.
        while True:
            # Lee un fotograma del video.
            ret, frame = cap.read()
            # Si no se pudo leer el fotograma (fin del video), rompe el bucle.
            if not ret:
                break

            # Realiza la inferencia utilizando el modelo YOLO en el fotograma actual.
            # - conf_threshold: umbral de confianza para filtrar detecciones.
            # - device: especifica si se usa GPU o CPU.
            results = model(frame, conf=conf_threshold, device=device, verbose=False)[0]
            # Extrae las coordenadas de las cajas delimitadoras (bounding boxes) en formato [x1, y1, x2, y2].
            boxes = results.boxes.xyxy.cpu().numpy()
            # Extrae las probabilidades de detección de cada objeto.
            confs = results.boxes.conf.cpu().numpy()
            # Extrae las clases detectadas (identificadores de objetos).
            classes = results.boxes.cls.cpu().numpy()

            # Inicializa una bandera para indicar si se detectó un UAV en el fotograma.
            uav_detected = False
            # Itera sobre cada detección (caja, confianza y clase).
            for box, conf, cls in zip(boxes, confs, classes):
                # Filtra las detecciones para considerar solo UAVs (clase 0).
                if int(cls) != 0:
                    continue
                # Marca que se detectó un UAV en el fotograma.
                uav_detected = True
                # Convierte las coordenadas de la caja delimitadora a enteros.
                x1, y1, x2, y2 = map(int, box)
                # Dibuja un rectángulo verde alrededor del UAV detectado en el fotograma.
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Crea una etiqueta con la probabilidad de detección y la dibuja cerca de la caja delimitadora.
                label = f"{conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 2)
                # Acumula la probabilidad de detección en sum_conf.
                sum_conf += conf
                # Incrementa el contador de detecciones totales.
                total_detections += 1
            
            # Si se detectó al menos un UAV en el fotograma, lo apunta
            if uav_detected:
                detected_frame_count += 1
            
            # escribo el fotograma aunque no hay detectiones
            writer.write(frame)
                
    # Libera los recursos del video y del escritor al finalizar el procesamiento.
    cap.release()
    writer.release()

    # Imprime un resumen del procesamiento del video.
    print(f"Finished processing {video_path}, wrote {detected_frame_count} UAV frames to {output_path}")

    # Devuelve estadísticas del procesamiento:
    # - detected_frame_count: número de fotogramas con UAVs detectados.
    # - sum_conf: suma de las probabilidades de detección.
    # - total_detections: número total de detecciones realizadas.
    return detected_frame_count, sum_conf, total_detections


def main():
    input_dir = Path("./videos")
    output_dir = Path("./videos/tuned_full")
    conf_threshold = 0.3

    if not torch.cuda.is_available():
        print("CUDA not found, using CPU...")
        device = 'cpu'
    else:
        device = 'cuda:0'

    # Carga el modelo directamente en GPU (device '0')
    #model = YOLO("yolo11n.pt")
    #model = YOLO("runs/drone/yolo11n_full/weights/best.pt")
    model = YOLO("drone/yolo11n_full/weights/best.pt")
    model.to(device)
    model.fuse()

    video_files = list(input_dir.glob("*.*"))
    total_detected_frames = 0
    total_sum_conf = 0.0
    total_conf_count = 0
    processed_videos = 0

    if output_dir.exists():
        print(f"Removing all contents of {output_dir}")
        for item in output_dir.iterdir():
            if item.is_file():
                item.unlink()
            else:
                shutil.rmtree(item)

    for video_path in video_files:
        if video_path.suffix.lower() not in [".mp4", ".avi", ".mov", ".mkv"]:
            continue
        output_path = output_dir / f"{video_path.stem}_detected.mp4"
        print(f"Procesando {video_path} -> {output_path}")
        dfc, sc, dc = process_video(video_path, output_path, model, conf_threshold, device)
        processed_videos += 1
        total_detected_frames += dfc
        total_sum_conf += sc
        total_conf_count += dc

    avg_conf = total_sum_conf / total_conf_count if total_conf_count else 0
    print(f"=== Summary ===")
    print(f"Total videos found: {len(video_files)}")
    print(f"Videos processed: {processed_videos}")
    print(f"Device used: {device}")
    print(f"Total frames with UAV detections: {total_detected_frames}")
    print(f"Average confidence of all UAV detections: {avg_conf:.2f}")

    # ------------------------------------

    model = YOLO('yolov8n.pt')           # nano weights (~5 MB)

    if SOURCE.isdigit():
        cap = cv2.VideoCapture(int(SOURCE), cv2.CAP_V4L2)
    else:
        cap = cv2.VideoCapture(SOURCE)

    if not cap.isOpened():
        raise SystemExit(f"❌  Could not open {SOURCE}")

if __name__ == "__main__":
    main()
