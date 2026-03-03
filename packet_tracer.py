import socket
import wave
import threading
import queue

HOST = "127.0.0.1"
PORT = 43001
WHISPER_PORT = 43002

# Queue to hand off audio chunks to the sender thread
send_q: "queue.Queue[bytes]" = queue.Queue(maxsize=200)
STOP = object()

def sender(sock: socket.socket):
    """Runs in a background thread, sends queued chunks."""
    while True:
        item = send_q.get()
        try:
            if item is STOP:
                return
            sock.sendall(item)  # blocks here, but only blocks this thread
        finally:
            send_q.task_done()


whisper_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
whisper_socket.connect((HOST, WHISPER_PORT))

# Start sender thread
t = threading.Thread(target=sender, args=(whisper_socket,), daemon=True)
t.start()

full_wav_file = wave.open("./audio/complete_file.wav", "wb")
full_wav_file.setnchannels(1)
full_wav_file.setsampwidth(16 // 8)
full_wav_file.setframerate(16000)

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    conn, addr = s.accept()

    with conn:
        print(f"Connect by {addr}")
        for i in range(100):
            data = conn.recv(65332)
            if not data:
                break

            # enqueue for async sending (won't block unless queue is full)
            try:
                send_q.put_nowait(data)
            except queue.Full:
                # Block sending until there is room to send 
                send_q.put(data)

            filename = f"./audio/file{i}.wav"
            wav_file = wave.open(filename, "wb")
            wav_file.setnchannels(1)
            wav_file.setsampwidth(16 // 8)
            wav_file.setframerate(16000)
            wav_file.writeframes(data)
            wav_file.close()

            full_wav_file.writeframes(data)

# clean shutdown
send_q.put(STOP)
send_q.join()
whisper_socket.close()
full_wav_file.close()

print("done.")