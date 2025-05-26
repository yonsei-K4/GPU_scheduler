import requests
import json
import time
import os

BASE_URL = "http://localhost:7071/api"
MODEL_FILE = os.path.join(os.path.dirname(__file__), "models.json")

def load_model_list():
    try:
        with open(MODEL_FILE, "r") as f:
            data = json.load(f)
            return data.get("models", [])
    except Exception as e:
        print(f"❌ 모델 파일 로딩 실패: {e}")
        return []
    
def post_loader():
    print("📦 [LOADER] 모델 로딩 요청 중...")
    res = requests.post(f"{BASE_URL}/loader", json={"model": "resnet50"})
    print(res.status_code)
    print(res.text)

def get_list():
    print("\n📋 [LIST] 로드된 모델 목록 요청 중...")
    res = requests.get(f"{BASE_URL}/list")
    print(res.status_code)
    print(res.text)

def post_inference(model_name="resnet50"):
    res = requests.post(f"{BASE_URL}/inference", json={"model": model_name})
    if(res.status_code != 200):
        print(res.status_code)
        print(res.text)

def get_profile():
    print("\n📊 [PROFILE] GPU 모니터링 요청 중...")
    res = requests.get(f"{BASE_URL}/profile")
    print(res.status_code)
    print(res.text)

def main():
    try:
        post_loader()
        time.sleep(1)

        model_list = load_model_list()
        if not model_list:
            print("❌ 모델 목록이 비어 있습니다.")
            return

        iter = 100
        for model in model_list * iter:
            post_inference(model)

    except Exception as e:
        print(f"❌ 예외 발생: {e}")

if __name__ == "__main__":
    main()
