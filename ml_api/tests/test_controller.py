import io
import json
import math
import os

#from neural_network_model.config import config as ccn_config
#from regression_model import __version__ as _version
#from regression_model.config import config as model_config
#from regression_model.processing.data_management import load_dataset

#from api import __version__ as api_version


def test_health_endpoint_returns_200(flask_test_client):
    # When
    response = flask_test_client.get('/health')

    # Then
    assert response.status_code == 200

def test_question_answering_works(flask_test_client):
    # Given
    # Load the test data from the regression_model package
    # This is important as it makes it harder for the test
    # data versions to get confused by not spreading it
    # across packages.
    post_json = '{ "sent1": "Where is my stuff?", "sent2": "Do you have my stuff?" }'

    # When
    response = flask_test_client.post('/v1/predict/question_answering',
                                      json=json.loads(post_json))

    # Then
    assert response.status_code == 200
    response_json = json.loads(response.data)
    prediction = response_json['predictions']
    response_version = response_json['version']
    assert math.ceil(prediction*1000) == 808
    assert response_version == '0'
