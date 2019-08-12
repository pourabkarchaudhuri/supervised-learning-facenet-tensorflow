import zerorpc
import os
from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename


from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename
import base64
import time

import urllib.request # For saving image as URL

dir_path = os.path.dirname(os.path.realpath(__file__))

UPLOAD_FOLDER = dir_path + "\\image_uploads"
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = set(['jpg', 'png', 'jpeg'])


#Example for Root Route
@app.route("/")
def index():
    return "Running!"


@app.route('/recognize', methods=['POST'])
def post_example():
 
    c = zerorpc.Client()
    c.connect("tcp://127.0.0.1:4242")
    print(request.headers)
    if not request.headers.get('Content-type') is None:
        if(request.headers.get('Content-type').split(';')[0] == 'multipart/form-data'):
            if 'image' in request.files.keys():
                file = request.files['image']
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                path = UPLOAD_FOLDER + "\\" + filename
                result = c.classifyFile(path)

                return jsonify({
                    "result": result
                })
        elif(request.headers.get('Content-type') == 'application/json'):
            if(request.data == b''):
                return jsonify(get_status_code("Invalid body", "Please provide valid format for Image")), 415
            else:
                body = request.get_json()
                if "url" in body.keys():
                    upload_path = UPLOAD_FOLDER + "\\local_" + str(time.time()) + ".jpg"
                    try:
                        urllib.request.urlretrieve(body['url'],upload_path)
                    except:
                        return jsonify(get_status_code("Invalid URL", "Cannot process image from given URL, Please provide valid URL.")), 415
                    
                    try:
                        result = c.classifyFile(upload_path)
                        return jsonify({
                            "result": result
                        })
                    except:
                        return jsonify(get_status_code("Unspecified", "Internal server error")), 500
                elif "image_string" in body.keys():
                    img_string = body['image_string']
                    try:
                        str_image = img_string.split(',')[1]
                        imgdata = base64.b64decode(str_image)
                        filename = UPLOAD_FOLDER + "\\image_file_" + str(time.time()) + ".jpg"
                        with open(filename, 'wb') as f:
                            f.write(imgdata)
                    except:
                        return jsonify(get_status_code("Invalid String", "Please provide valid image_string")), 415
                    try:
                        result = c.classifyFile(filename)
                        return jsonify({
                            "result": result
                        }), 200
                    except:
                        return jsonify(get_status_code("Unspecified", "Internal server error")), 500

                    # return "Keyword URL not found"
        else:
            return jsonify(get_status_code("Invalid header", "Please provide correct header with correct data")), 415
           
    else:
        return jsonify(get_status_code("Invalid Header", "Please provide valid header")), 401


def get_status_code(argument, message):
    res = {
        "error": {
            "code": argument,
            "message": message
        }
    }
    return res
if __name__ == "__main__":
    app.run(debug=True, port=5000)