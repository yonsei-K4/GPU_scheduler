import azure.functions as func
import logging
import requests
from profile.runner import ModelRunner

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

@app.route(route="test")
def test(req: func.HttpRequest) -> func.HttpResponse:
    return func.HttpResponse(f"Test function is working.")

@app.route(route="main")
def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    # Call the 'test' function
    test_function_url = req.url.replace("main", "test")  # Adjust URL for 'test' function
    response = requests.get(test_function_url)

    return func.HttpResponse(f"Response from 'test': {response.text}")
        
@app.route(route="resnet18")
def resnet18(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('resnet18 function processed a request.')

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
        "0": {0: 128},
        "1": {1: 128},
        "2": {2: 128},
        "3": {3: 128}
    }
    ModelRunner(model_list, runtime_info).run_all()

    return func.HttpResponse(f"Resnet18 model run successfully")

    