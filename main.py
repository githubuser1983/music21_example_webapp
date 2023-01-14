import os
from flask import Flask, flash, request, redirect, url_for,render_template
from werkzeug.utils import secure_filename
from bezier_with_chords import processMidi
from connected_components import processFile
import os, io, zipfile, time

from flask import send_file, make_response


UPLOAD_FOLDER = './static/uploads'
ALLOWED_EXTENSIONS = {'mid', 'musicxml'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and any([ filename.endswith("."+ext) for ext in ALLOWED_EXTENSIONS])

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        file_length = file.seek(0, os.SEEK_END)
        # also can use tell() to get current position
        # file_length = file.tell()
        # seek back to start position of stream, 
        # otherwise save() will write a 0 byte file
        # os.SEEK_END == 0
        file.seek(0, os.SEEK_SET)
        if file_length > 1500*1024:
            #flash("File to large")
            return '''Error: File larger then 5kB'''
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            eps = float(request.form["eps"])
            if not eps in [0.1,0.01,0.001,0.0001]:
                return None            
            filename = secure_filename(file.filename)
            savefilename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(savefilename)
            print(int(request.form["tempo"]))
            print(request.form["funcType"])
            title = request.form["title"]
            author = request.form["composer"]
            fileType = request.form["fileType"]
            if not fileType in ["mid","musicxml"]:
                return None
            #processedFile = processMidi(file = savefilename,tempo=int(request.form["tempo"]),funcType=request.form["funcType"])
            processedFile = processFile(file = savefilename,tempo=int(request.form["tempo"]),funcType=request.form["funcType"],eps=eps,title=title,author=author,fileType=fileType)
            print(processedFile)
            return redirect(url_for('download_file', name=processedFile.split("/")[-1]))
    return render_template("upload.html")
    
from flask import send_from_directory

@app.route('/uploads/<name>')
def download_file(name):
    #return send_from_directory(app.config["UPLOAD_FOLDER"], name)    
    fileobj = io.BytesIO()
    filepath = os.path.join(app.config["UPLOAD_FOLDER"],name)
    print(filepath)
    with zipfile.ZipFile(fileobj, 'w') as zip_file:
        zip_info = zipfile.ZipInfo(filepath)
        zip_info.date_time = time.localtime(time.time())[:6]
        zip_info.compress_type = zipfile.ZIP_DEFLATED
        with open(filepath, 'rb') as fd:
            zip_file.writestr(zip_info, fd.read())
    fileobj.seek(0)
    response = make_response(fileobj.read())
    response.headers.set('Content-Type', 'zip')
    response.headers.set('Content-Disposition', 'attachment', filename='%s.zip' % os.path.basename(filepath))
    return response

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8001)    
