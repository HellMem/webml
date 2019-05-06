from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.renderers import JSONRenderer
from rest_framework.parsers import JSONParser

from .ml.iris_classifier import *
from .sagemaker.iris_classifier_sm import *


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
        sepal_length = float(request.GET.get('sepal_length', '0.0'))
        sepal_width = float(request.GET.get('sepal_width', '0.0'))
        petal_length = float(request.GET.get('petal_length', '0.0'))
        petal_width = float(request.GET.get('petal_width', '0.0'))

        classes = predict_labels(sepal_length, sepal_width, petal_length, petal_width)
        #sm_classes = iris_prediction(sepal_length, sepal_width, petal_length, petal_width)

        response = {}
        response['Pred'] = classes
        #response['PredSM'] = sm_classes

        return JSONResponse(response)
    elif request.method == 'POST':
        message = {"message": "Training done"}
        return JSONResponse(message)
