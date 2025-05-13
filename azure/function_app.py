import azure.functions as func
import logging
import requests
# from profile.runner import ModelRunner
from session import SessionManager

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

@app.route(route="inference")
def inference(req: func.HttpRequest) -> func.HttpResponse:
    '''ModelRunner를 통해 모델을 실행하는 함수입니다.'''
    logging.info('inferencing...')
    

    return func.HttpResponse(f"inference finished.")

@app.route(route="list")
def list_models(req: func.HttpRequest) -> func.HttpResponse:
    '''모델 목록을 반환하는 함수입니다.'''
    response = "list models...\n"
    logging.info('list models...')
    
    for model_name in SessionManager.sessList.keys():
        response += f"Model {model_name} is loaded.\n"
        logging.info(f"Model {model_name} is loaded.")
    
    # 모델 목록을 가져오는 로직을 여기에 추가합니다.
    # 예를 들어, 모델 이름과 경로를 반환하는 등의 작업을 수행할 수 있습니다.

    response += "list models finished."
    return func.HttpResponse(response)

@app.route(route="profile")
def monitor_gpu(req: func.HttpRequest) -> func.HttpResponse:
    '''GPU 모니터링을 위한 함수입니다.'''
    logging.info('monitoring gpu...')
    
    # GPU 모니터링 로직을 여기에 추가합니다.
    # 예를 들어, GPU 사용량을 기록하거나 그래프를 생성하는 등의 작업을 수행할 수 있습니다.

    return func.HttpResponse(f"GPU monitoring finished.")

@app.route(route="loader")
def loader(req: func.HttpRequest) -> func.HttpResponse:
    '''모델을 로드하는 함수입니다.'''
    response = "loader processed...\n"
    logging.info('loader processed.')

    try:
        req_body = req.get_json()
    except ValueError:
        pass
    else:
        model= req_body.get('model')
    
    model_list = ["alexnet", "resnet18", "resnet50", "resnext101"]
    providers = {"alexnet": ("CUDAExecutionProvider", {"device_id": 0}),
                 "resnet18": ("CUDAExecutionProvider", {"device_id": 1}),
                  "resnet50": ("CUDAExecutionProvider", {"device_id": 2}),
                    "resnext101": ("CUDAExecutionProvider", {"device_id": 3})
                }

    for model_name in model_list:
        session = SessionManager.fetch_model(model_name, providers[model_name])
        if session is None:
            return func.HttpResponse(f"Failed to load model {model_name}", status_code=500)
        else:
            logging.info(f"Model {model_name} loaded successfully")
            response += f"Model {model_name} loaded successfully\n"
    
    response += "Model loading finished.\n"
    logging.info('Model loading finished.')
    return func.HttpResponse(response)

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
    
    model_name = req.params.get('model')
    if not model_name:
        return func.HttpResponse("Please pass a model name in the query string or in the request body", status_code=400)
    if model_name not in SessionManager.sessList.keys():
        return func.HttpResponse(f"Model {model_name} is not loaded", status_code=404)
    
    response = SessionManager.run_model(model_name)
   
    logging.info(response)
    return func.HttpResponse(response)

@app.route(route="shutdown")
def shutdown(req: func.HttpRequest) -> func.HttpResponse:
    """Shutdown the local server."""
    import sys
    sys.exit(0)  # Forcefully exit the process
    return func.HttpResponse("Server shutting down...")