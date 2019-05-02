from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.renderers import JSONRenderer
from rest_framework.parsers import JSONParser


class JSONResponse(HttpResponse):
    """
    An HttpResponse that renders its content into JSON.
    """

    def __init__(self, data, **kwargs):
        content = JSONRenderer().render(data)
        kwargs['content_type'] = 'application/json'
        super(JSONResponse, self).__init__(content, **kwargs)


@csrf_exempt
def iris_model(request):
    """
    List all code serie, or create a new serie.
    """
    if request.method == 'GET':
        q = request.GET.get('q', '')
		qw = request.GET.get('qw', '')
        cosa = {}
        cosa["cosilla"] = "cosilla"
        cosa["12"] = "12"
        cosa["arbol"] = "arbol"
        cosa["q"] = q
		cosa["qw"] = qw

        return JSONResponse(cosa)
    elif request.method == 'POST':
        message = {"message": "Training done"}
        return JSONResponse(message)
