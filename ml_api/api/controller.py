from api.validation import validate_inputs
from flask import Blueprint, request, jsonify
from glue_models import glue_predict as pr

prediction_app = Blueprint('prediction_app', __name__)
import os
#print(os.environ['PWD'])

ft=pr.FrameworkPredictor('../')

@prediction_app.route('/health', methods = ['GET'])
def health():
    if request.method == 'GET':
        return 'ok'

@prediction_app.route('/nice', methods = ['GET'])
def nice():
    if request.method == 'GET':
        return str(1+1)

@prediction_app.route('/v1/predict/question_answering', methods = ['POST'])
def question_answering():
    if request.method == 'POST':
        
        
        # Step 1: Extract POST data from request body as JSON
        json_data = request.get_json()
        #_logger.debug(f'Inputs: {json_data}')

        # Step 2: Validate the input using marshmallow schema
        sent1, sent2, errors = validate_inputs(input_data=json_data)

        # Step 3: Model prediction
        result = ft.question_answering(sent1, sent2)
        #_logger.debug(f'Outputs: {result}')

        # Step 4: Convert numpy ndarray to list
        predictions = result#.get('predictions').tolist()
        version = '0' #result.get('version')

        # Step 5: Return the response as JSON
        return jsonify({'predictions': predictions,
                        'version': version,
                        'errors': errors})

@prediction_app.route('/v1/predict/question_similarity', methods = ['POST'])
def question_similarity():
    if request.method == 'POST':
        # Step 1: Extract POST data from request body as JSON
        json_data = request.get_json()
        #_logger.debug(f'Inputs: {json_data}')

        # Step 2: Validate the input using marshmallow schema
        sent1, sent2, errors = validate_inputs(input_data=json_data)

        # Step 3: Model prediction
        result = ft.question_similarity(sent1, sent2)
        #_logger.debug(f'Outputs: {result}')

        # Step 4: Convert numpy ndarray to list
        predictions = result#.get('predictions').tolist()
        version = '0' #result.get('version')

        # Step 5: Return the response as JSON
        return jsonify({'predictions': predictions,
                        'version': version,
                        'errors': errors})