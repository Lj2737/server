import requests
with open('E:/Project-python/chat2MCU/audio/input/sop1.wav', 'rb') as f:
    resp = requests.post(
        'http://127.0.0.1:8090/api/v1/gateway/behavior-recognition',
        files={'audio_file': f},
        data={
            'device_no': 'BADGE0001',
            'event_time': '2026-05-08 10:30:00',
            'request_id': 'req-test-001'
        }
    )
print(resp.status_code, resp.json())