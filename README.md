# Mining Site Vehicle Counter

Sistem computer vision untuk menghitung kendaraan (haul truck & kendaraan lain) di situs pertambangan menggunakan **YOLO** + **ByteTrack**. Tersedia dalam mode **CLI** (single video) dan **Web Server** (multi-kamera dengan dashboard live).

## Fitur

- Deteksi dan tracking kendaraan dengan YOLO + ByteTrack
- Menghitung kendaraan yang melintasi garis virtual (counting line)
- Dua kelas: `haul_truck` dan `other_vehicles`
- Web dashboard real-time dengan MJPEG stream + Server-Sent Events
- Multi-kamera (konfigurasi via YAML)
- Auto-reconnect untuk RTSP streams
- Start/Stop & Reset counter per kamera dari dashboard
- Loop video untuk file (berguna untuk testing)

## Struktur Project

```
vehicle-counter-main/
├── counter.py              # CLI single-video counter
├── server.py               # Web server multi-kamera
├── vcmodel1.onnx           # Model YOLO ONNX (pre-trained)
├── bytetrack.yaml          # Konfigurasi ByteTrack tracker
├── cameras.yaml            # Konfigurasi kamera & model
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker build
├── docker-compose.yml      # Docker compose
├── run_server.sh           # Script untuk menjalankan server
├── videos/                 # Folder untuk file video input
│   └── .gitkeep
└── templates/
    └── dashboard.html      # Frontend dashboard
```

## Prerequisites

- **Model ONNX**: File `vcmodel1.onnx` sudah tersedia di project ini
- **Video/RTSP**: Letakkan file `.mp4` di folder `videos/`, atau gunakan URL RTSP

### CPU vs GPU

| | CPU | GPU (CUDA) |
|--|-----|------------|
| **onnxruntime** | `onnxruntime` | `onnxruntime-gpu` |
| **PyTorch index** | `https://download.pytorch.org/whl/cpu` | PyPI default (CUDA) |
| **Image size** | ~1.5 GB | ~3.5 GB |
| **Kinerja** | Lambat (real-time possible utk 1-2 kamera @ rendah res) | Cepat |

> **VPS tanpa GPU**: Jalankan saja langsung. Kode auto-detect device — tidak perlu modifikasi kode apapun.

## Instalasi

### Lokal (tanpa Docker)

1. **Buat virtual environment** (opsional tapi disarankan):
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Atur cameras.yaml**: Sesuaikan URL video/RTSP dan label kamera. Path video relatif ke folder `videos/`

### Docker

1. **Build image**:
   ```bash
   docker compose build
   ```

2. **Siapkan volume**:
   - File model `vcmodel1.onnx` sudah otomatis di-mount
   - Letakkan file video `.mp4` di `./videos/`
   - Edit `cameras.yaml` jika perlu

3. **Jalankan**:
   ```bash
   docker compose up
   ```

   Atau build + jalan sekaligus:
   ```bash
   docker compose up --build
   ```

   Dashboard bisa diakses di `http://localhost:5000`

## Penggunaan

### CLI Single Video

```bash
python counter.py --video path/to/video.mp4 --show
```

| Argumen | Deskripsi |
|---------|-----------|
| `--video` | Path ke video atau URL RTSP **(wajib)** |
| `--model` | Path ke model ONNX (default: `vcmodel1.onnx`) |
| `--tracker` | Path ke config tracker (default: `bytetrack.yaml`) |
| `--output` | Nama file output (default: `output_counted.mp4`) |
| `--conf` | Confidence threshold (default: 0.25) |
| `--show` | Tampilkan jendela preview |

### Web Server Multi-Kamera

```bash
python server.py --config cameras.yaml --port 5000
```

Atau via script:
```bash
./run_server.sh
./run_server.sh --config cameras.yaml --port 8080
```

Kemudian buka browser di `http://localhost:5000`

## Konfigurasi

### cameras.yaml

```yaml
model: vcmodel1.onnx           # File model YOLO ONNX yang sudah ada
tracker: bytetrack.yaml        # Path ke config ByteTrack
conf: 0.25                     # Confidence threshold

show_classes:
  - haul_truck                 # Kelas yang ditampilkan
  # - other_vehicles           # Uncomment untuk menampilkan kelas ini

host: 0.0.0.0
port: 5000

cameras:
  - url:   videos/gate_a.mp4   # Ganti dengan path video (relatif dari project root) atau RTSP
    label: "Gate A"            # Nama kamera
    line_x: 0.5                # Posisi garis (0.0-1.0, relatif terhadap lebar frame)
    loop:  true                # Loop video (false untuk RTSP)
```

### bytetrack.yaml

| Parameter | Default | Deskripsi |
|-----------|---------|-----------|
| `track_high_thresh` | 0.25 | Threshold high untuk tracking |
| `track_low_thresh` | 0.05 | Threshold low untuk tracking |
| `new_track_thresh` | 0.30 | Threshold track baru |
| `track_buffer` | 60 | Frame buffer untuk lost track (~2 detik @30fps) |
| `match_thresh` | 0.85 | Threshold matching |
| `fuse_score` | True | Fusion score |

## Cara Kerja Counting

1. Garis vertikal digambar di posisi `line_x` (default: tengah frame)
2. Setiap kendaraan yang terdeteksi dilacak dengan ByteTrack (track ID unik)
3. Saat center X dari bounding box melewati garis, kendaraan dihitung
4. Minimal pergerakan 60px dari posisi pertama diperlukan untuk mencegah false positive

## Catatan

- Untuk konfigurasi dengan kredensial RTSP (`rtsp://user:password@ip/stream`), simpan sebagai `cameras.local.yaml` (file ini sudah di gitignore)
- File model `*.onnx` sudah di gitignore - jangan commit model ke repository
- **CPU**: Aplikasi auto-detect device — tidak ada perubahan kode diperlukan
- **GPU**: Ganti `onnxruntime` → `onnxruntime-gpu` di `requirements.txt`, dan untuk Docker tambahkan `runtime: nvidia` di `docker-compose.yml`
- Di Docker, mount folder video dan model sebagai volume untuk fleksibilitas
