#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS2 /fix( NavSatFix )를 받아서 경로(polyline)를 누적·표시하는 초경량 웹 뷰어.
- 브라우저에 Leaflet 지도를 띄우고, 마우스 커서를 경로 위에 올리면 해당 지점의 위도/경도 표시
- Reset 버튼으로 서버 메모리에 저장된 경로 즉시 삭제
- 외부 의존성: Flask (웹), Leaflet(클라이언트 CDN). rosbridge 불필요, 단일 파일.

실행:
  1) (ROS2 환경 활성화)  source /opt/ros/<distro>/setup.bash && source ~/ros2_ws/install/setup.bash
  2) (필요시) pip install flask
  3) python3 gps_path_viewer.py --port 5000 --min-step-m 1.0 --max-points 5000
  4) 브라우저에서  http://<호스트IP>:5000  접속

옵션:
  --port           : 웹서버 포트 (기본 5000)
  --min-step-m     : 점 간 최소 간격(미터). 더 촘촘히 그리고 싶으면 0.2~0.5로 내림
  --max-points     : 서버에 메모리 저장 최대 점 수 (오래 달려도 메모리 폭주 방지)
  --downsample     : 클라이언트로 보낼 때 샘플링 간격(정수). 1=전체, 3=3개당 1개
  --topic          : NavSatFix 토픽 이름 (기본 /fix)
  --bind           : 바인드 주소 (기본 0.0.0.0)

참고:
  - 오프라인에서 타일이 필요하면 Leaflet 대신 로컬 타일서버나 XYZ 캐시로 교체 가능
  - RViz 대안: /fix → nav_msgs/Path 변환 후 RViz로 시각화. 단, 마우스 hover 위경도 표시/초경량 웹 UI는 본 스크립트가 유리
"""

import argparse
import math
import threading
import time
from typing import List, Tuple

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix

from flask import Flask, jsonify, request, Response

# =============================
# 1️⃣ 유틸: 거리 계산(Haversine)
# =============================
EARTH_R = 6371000.0  # meters


def haversine_m(lat1, lon1, lat2, lon2):
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return EARTH_R * c


# ============================================
# 2️⃣ ROS2 노드: /fix 구독 → 경로 리스트 누적
# ============================================
class GpsPathNode(Node):
    def __init__(self, topic: str, min_step_m: float, max_points: int):
        super().__init__("gps_path_viewer")
        self.topic = topic
        self.min_step_m = float(min_step_m)
        self.max_points = int(max_points)

        self.path: List[Tuple[float, float]] = []  # [(lat, lon), ...]
        self._lock = threading.RLock()
        self._last_latlon = None

        self.create_subscription(NavSatFix, self.topic, self._cb_fix, 10)
        self.get_logger().info(f"Subscribed to {self.topic}")

    def _cb_fix(self, msg: NavSatFix):
        lat = float(msg.latitude)
        lon = float(msg.longitude)
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            return
        with self._lock:
            if self._last_latlon is None:
                self.path.append((lat, lon))
                self._last_latlon = (lat, lon)
            else:
                lat0, lon0 = self._last_latlon
                dist = haversine_m(lat0, lon0, lat, lon)
                if dist >= self.min_step_m:
                    self.path.append((lat, lon))
                    self._last_latlon = (lat, lon)
            # 메모리 안전장치
            if len(self.path) > self.max_points:
                overflow = len(self.path) - self.max_points
                del self.path[0:overflow]

    # 외부에서 안전하게 읽기
    def get_path_copy(self):
        with self._lock:
            return list(self.path)

    def reset_path(self):
        with self._lock:
            self.path.clear()
            self._last_latlon = None


# =============================
# 3️⃣ Flask 앱: 초경량 API & UI
# =============================
app = Flask(__name__)
NODE: GpsPathNode = None  # 런타임 주입

INDEX_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>NALSEM GPS Path Viewer</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" crossorigin="" />
  <style>
    html, body, #map { height: 100%; margin: 0; }
    .topbar { position: absolute; top: 10px; right: 10px; z-index: 1000; background: rgba(255,255,255,0.9); padding: 8px 10px; border-radius: 8px; font-family: sans-serif; box-shadow: 0 2px 8px rgba(0,0,0,0.15); }
    .coord { font-size: 12px; color: #333; }
    button { border: 0; padding: 6px 10px; border-radius: 6px; cursor: pointer; }
  </style>
</head>
<body>
  <div id="map"></div>
  <div class="topbar">
    <button id="resetBtn">Reset</button>
    <label style="margin-left:8px;"><input type="checkbox" id="followChk" checked /> Follow</label>
    <span class="coord" id="coord">–</span>
  </div>

  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" crossorigin=""></script>
  <script>
    const map = L.map('map');
    const osm = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', { maxZoom: 22, attribution: '&copy; OpenStreetMap' });
    osm.addTo(map);

    const line = L.polyline([], { weight: 4 }).addTo(map);
    const head = L.circleMarker([0,0], { radius: 5 }).addTo(map);

    const coordEl = document.getElementById('coord');
    const followChk = document.getElementById('followChk');

    // 경로 위에서 마우스 움직일 때, 현재 커서 위치(lat,lng) 표시
    line.on('mousemove', (e) => {
      const { lat, lng } = e.latlng;
      coordEl.textContent = `${lat.toFixed(6)}, ${lng.toFixed(6)}`;
    });
    // 라인에서 벗어나면 지움
    line.on('mouseout', () => { coordEl.textContent = '–'; });

    async function fetchPath() {
      try {
        const res = await fetch('/data');
        const js = await res.json();
        const path = js.path;
        if (!path || path.length === 0) return;
        line.setLatLngs(path);
        const last = path[path.length - 1];
        head.setLatLng(last);
        if (!map._zoom) {
          map.setView(last, 17);
        } else if (followChk.checked) {
          map.panTo(last, { animate: true });
        }
      } catch (e) { /* 네트워크 불가 시 무시 */ }
    }

    // 1초마다 경로 갱신(가벼운 폴링)
    setInterval(fetchPath, 1000);

    // Reset 버튼: 서버 메모리 지우기
    document.getElementById('resetBtn').onclick = async () => {
      await fetch('/reset', { method: 'POST' });
      line.setLatLngs([]);
      coordEl.textContent = '–';
    };

    // 최초 1회 로드
    fetchPath();
  </script>
</body>
</html>
"""


@app.get("/")
def index():
    return Response(INDEX_HTML, mimetype="text/html")


@app.get("/data")
def data():
    ds = int(request.args.get("downsample", app.config.get("DOWNSAMPLE", 1)))
    path = NODE.get_path_copy() if NODE else []
    if ds > 1 and len(path) > ds:
        path = path[::ds]
        if path[-1] != NODE.get_path_copy()[-1]:
            path.append(NODE.get_path_copy()[-1])  # 마지막점 보존
    return jsonify({"path": path, "count": len(path)})


@app.post("/reset")
def reset():
    if NODE:
        NODE.reset_path()
    return jsonify({"ok": True})


# ===================================
# 4️⃣ 메인: ROS2 + Flask 동시 구동
# ===================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--bind", type=str, default="0.0.0.0")
    parser.add_argument("--topic", type=str, default="/fix")
    parser.add_argument("--min-step-m", type=float, default=1.0)
    parser.add_argument("--max-points", type=int, default=5000)
    parser.add_argument("--downsample", type=int, default=1)
    args = parser.parse_args()

    app.config["DOWNSAMPLE"] = max(1, int(args.downsample))

    rclpy.init()
    node = GpsPathNode(args.topic, args.min_step_m, args.max_points)

    global NODE
    NODE = node

    # ROS2 스레드: spin
    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()

    # Flask 메인 스레드 실행
    try:
        app.run(host=args.bind, port=args.port, threaded=True)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
