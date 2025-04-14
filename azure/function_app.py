import azure.functions as func
import logging
import requests
from profile.runner import ModelRunner
from time import sleep

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

@app.route(route="inference")
def inference(req: func.HttpRequest) -> func.HttpResponse:
    '''ModelRunner를 통해 모델을 실행하는 함수입니다.'''
    logging.info('inferencing...')
    
    ModelRunner.run_all()

    return func.HttpResponse(f"inference finished.")


@app.route(route="profile")
def monitor_gpu(req: func.HttpRequest) -> func.HttpResponse:
    '''GPU 모니터링을 위한 함수입니다.'''
    logging.info('monitoring gpu...')
    
    # GPU 모니터링 로직을 여기에 추가합니다.
    # 예를 들어, GPU 사용량을 기록하거나 그래프를 생성하는 등의 작업을 수행할 수 있습니다.

    return func.HttpResponse(f"GPU monitoring finished.")

@app.route(route="runner")
def runner(req: func.HttpRequest) -> func.HttpResponse:
    '''전체적인 runner'''
    logging.info('runner processed.')

    valid = req.params.get('valid')
    if not valid:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            model= req_body.get('valid')
    
    model_list = ["../profile/resnet18.onnx",
                  "../profile/resnet50.onnx",
                  "../profile/alexnet.onnx",
                  "../profile/vgg19.onnx"]
    runtime_info = {
        "0": {0: 128, 1: 128},
        # "1": {1: 128},
        # "2": {2: 128},
        # "3": {3: 128}
    }

    ModelRunner(model_list, runtime_info)
    sleep(5)

    # Call the 'test' function
    inference_func_url = req.url.replace("runner", "inference")  # Adjust URL for 'test' function
    response = requests.get(inference_func_url)

    return func.HttpResponse(f"Resnet18 model run successfully")
