import azure.functions as func
import logging
from profile.runner import ModelRunner

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

@app.route(route="main")
def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    model = req.params.get('model')
    if not model:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            model= req_body.get('model')
    
    if model:
        pass
        
@app.route(route="resnet18")
def resnet18(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    valid = req.params.get('valid')
    if not valid:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            model= req_body.get('valid')
    
    model_list = ["../profile/resnet18.onnx"]
    runtime_info = {
        "0": {0: 32},
    }
    ModelRunner(model_list, runtime_info).run_all()

    return func.HttpResponse(f"Resnet18 model run successfully")

    