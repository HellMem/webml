from .responses import iris_model
from django.views.decorators.csrf import csrf_exempt


@csrf_exempt
def iris(request):
    return iris_model(request)
