import azure.functions as func
import logging
from runner import runner

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
        
