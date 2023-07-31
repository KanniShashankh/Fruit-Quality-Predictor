import os
import sys
import numpy as np
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from PIL import Image, ImageFile
import my_tf_mod
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/Prediction', methods=['GET','POST'])
def pred():
    try:
        if request.method=='POST':
            file = request.files['file']
            org_img, img = my_tf_mod.preprocess(file)
            fruit_dict = my_tf_mod.classify_fruit(img)
            rotten = my_tf_mod.check_rotten(img)

            # Create a BytesIO object to store the plot image temporarily
            img_x = BytesIO()

            # Plot the original image and save it to the BytesIO object
            plt.imshow(org_img / 255.0)
            plt.axis('off')
            plt.savefig(img_x, format='png')
            plt.close()
            # Seek to the beginning of the BytesIO object
            img_x.seek(0)

            # Convert the image to base64 encoding
            plot_url = base64.b64encode(img_x.getvalue()).decode('utf-8')
            return render_template('Pred3.html', fruit_dict=fruit_dict, rotten=rotten, plot_url=plot_url)
    except Exception as e:
        print("THIS IS THE ERROR " , e)
        return render_template('Pred3.html', fruit_dict=fruit_dict, rotten=rotten, plot_url=None)




if __name__=='__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))