import socket

# --- Configuración ---
UDP_IP = "0.0.0.0"  # Escuchar en todas las interfaces de red
UDP_PORT = 5600     # Puerto UDP de OpenHD
BUFFER_SIZE = 1024

# --- Creación del Socket ---
try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # --- LA LÍNEA CLAVE ---
    # Esta línea le dice al sistema operativo que permita que otras
    # aplicaciones también usen este puerto.
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    sock.bind((UDP_IP, UDP_PORT))
    print(f"✅ Escuchando en el puerto UDP COMPARTIDO {UDP_PORT}...")

except socket.error as e:
    print(f"❌ Error al crear o enlazar el socket: {e}")
    exit()

# --- Bucle de Escucha ---
try:
    while True:
        data, addr = sock.recvfrom(BUFFER_SIZE)
        print(f"📦 ¡Paquete recibido de {addr}! Tamaño: {len(data)} bytes")
except KeyboardInterrupt:
    print("\n🛑 Programa terminado por el usuario.")
finally:
    sock.close()
    print("Socket cerrado.")