import requests
import json



def iris_prediction(sepal_length, sepal_width, petal_length, petal_width):
    # api-endpoint
    URL = "https://fo9soej9w9.execute-api.us-east-1.amazonaws.com/Test"

    # data to be sent to api
    data_values = '{}, {}, {}, {}'.format(sepal_length, sepal_width, petal_length, petal_width)
    data = {'data': data_values}
    # data = {}

    headers = {'content-type': 'application/json'}
    # sending post request and saving response as response object
    r = requests.post(url=URL, data=json.dumps(data), headers=headers)

    return r._content


'''
if __name__ == "__main__":
    print(iris_prediction(6.4, 3.2, 4.5, 1.5))

'''
