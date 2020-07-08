#!/usr/bin/python
from flask import Flask
from flask_restx import Api, Resource, fields
from sklearn.externals import joblib
from Msencillo import valores


app = Flask(__name__)

api = Api(
    app, 
    version='1.0', 
    title='Predicciones genero de los plot',
    description='Predicción Plot')

ns = api.namespace('predict', 
     description='Phishing Classifier')
   
parser = api.parser()

parser.add_argument(
    'Plot', 
    type=str, 
    required=True, 
    help='Plot', 
    location='args')

resource_fields = api.model('Resource', {
    'result': fields.String,
})

@ns.route('/')
class PhishingApi(Resource):

    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()
        
       
        return {
         "result": valores(args['Plot'])}, 200

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
