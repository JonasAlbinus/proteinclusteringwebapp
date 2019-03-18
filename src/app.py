import os
import threading

import matplotlib
from flask import Flask, flash, redirect, render_template, request, url_for, send_from_directory
from clustering import cluster
from compute_analysis import compute_analysis
from pathlib import Path


app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


@app.route('/')
def index():
    return 'index'


@app.route('/home')
def home():
    return render_template('home.html', profile_image="static/AlbertSilvana.png")


@app.route('/resume')
def resume():
    return render_template('resume.html')


@app.route('/publications')
def publications():
    return render_template('publications.html')


@app.route('/clustering', methods=['GET', 'POST'])
def clustering():
    return render_template('clustering.html',
                           data=[{'name': '2'}, {'name': '3'}, {'name': '4'}, {'name': '5'}, {'name': '6'},
                                 {'name': '8'}])


@app.route("/test", methods=['GET', 'POST'])
def test():
    select = request.form.get('comp_select')
    cluster(select)
    return render_template('clustering.html',
                           data=[{'name': '2'}, {'name': '3'}, {'name': '4'}, {'name': '5'}, {'name': '6'},
                                 {'name': '8'}],
                           user_image="static/plots/plot.png")


@app.route("/testproteins", methods=['GET', 'POST'])
def testproteins():
    step = request.form.get('comp_select')
    protein = request.form.get('protein_select')
    alg = request.form.get('alg_select')
    algName = ""
    k = ""

    if int(step) == 2500:
        k = "4"
    elif int(step) == 1000:
        k = "10"
    elif int(step) == 500:
        k = "20"
    elif int(step) == 400:
        k = "25"
    elif int(step) == 150:
        k = "40"
    elif int(step) == 100:
        k = "100"
    elif int(step) == 50:
        k = "200"

    if alg == "K-Means Clustering":
        filename = ("static/data/" + protein + "_" + k + "_k-means" + ".png")
        file = protein + "_" + k + "_k-means" + ".png"
        algName = "k-means"
    elif alg == "Agglomerative Clustering":
        filename = ("static/data/" + protein + "_" + k + "_agg" + ".png")
        file = protein + "_" + k + "_agg" + ".png"
        algName = "agg"
    elif alg == "Birch Clustering":
        filename = ("static/data/" + protein + "_" + k + "_birch" + ".png")
        file = protein + "_" + k + "_birch" + ".png"
        algName = "birch"
    else:
        filename = ("static/data/" + protein + "_" + k + "_skfuzzy" + ".png")
        file = protein + "_" + k + "_skfuzzy" + ".png"
        algName = "skfuzzy"

    print("Searching for: " + filename)
    my_file = Path(filename)

    if my_file.is_file():
        return render_template('proteins.html', alg=alg, protein=protein, cluster_no=step,
                               data=get_data_array(step),
                               proteins=get_proteins_array(protein),
                               method=get_alg_array(alg),
                               user_image=filename,
                               filename=file
                               )
    else:
        compute_thread = threading.Thread(target=compute_analysis, args=(protein, int(step), algName))
        compute_thread.start()
        return render_template('proteins.html', alg=alg, protein=protein, cluster_no=step,
                               data=get_data_array(step),
                               proteins=get_proteins_array(protein),
                               method=get_alg_array(alg),
                               user_image="static/Refresh.png"
                               )


@app.route('/proteins', methods=['GET', 'POST'])
def proteins():
    return render_template('proteins.html',
                           data=get_data_array(50),
                           proteins=get_proteins_array("1GO1"),
                           method=get_alg_array('K-Means Clustering')
                           )


@app.route('/som', methods=['GET', 'POST'])
def som():
    return render_template('som.html',
                           proteins=[{'name': '7 Proteins'}, {'name': '58 Proteins'}])


@app.route('/testsom', methods=['GET', 'POST'])
def testsom():
    alg = request.form.get('protein_select')
    if alg == "7 Proteins":
        filename = "static/som/7Proteins.png"
    else:
        filename = "static/som/58Proteins.png"
    return render_template('som.html',
                           proteins=[{'name': '7 Proteins'}, {'name': '58 Proteins'}],
                           user_image=filename,
                           )


@app.route('/files', methods=['POST'])
def files():
    """Download a file."""
    filename = request.form.get('filename')
    ROOT_DIRECTORY = 'static/data/'
    return send_from_directory(ROOT_DIRECTORY, filename, as_attachment=True)


# No caching at all for API endpoints.
@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response


def get_proteins_array(protein):
    proteinsArray = []

    if protein == '1GO1':
        proteinsArray.append({'name': '1GO1', "Selected": True})
    else:
        proteinsArray.append({'name': '1GO1', "Selected": False})

    if protein == '1JT8':
        proteinsArray.append({'name': '1JT8', "Selected": True})
    else:
        proteinsArray.append({'name': '1JT8', "Selected": False})

    if protein == '1P1L':
        proteinsArray.append({'name': '1P1L', "Selected": True})
    else:
        proteinsArray.append({'name': '1P1L', "Selected": False})

    if protein == '1L3P':
        proteinsArray.append({'name': '1L3P', "Selected": True})
    else:
        proteinsArray.append({'name': '1L3P', "Selected": False})

    if protein == '4CG1':
        proteinsArray.append({'name': '4CG1', "Selected": True})
    else:
        proteinsArray.append({'name': '4CG1', "Selected": False})

    if protein == '6EQE':
        proteinsArray.append({'name': '6EQE', "Selected": True})
    else:
        proteinsArray.append({'name': '6EQE', "Selected": False})

    return proteinsArray


def get_alg_array(alg):
    alg_array = []

    if alg == 'K-Means Clustering':
        alg_array.append({'name': 'K-Means Clustering', "Selected": True})
    else:
        alg_array.append({'name': 'K-Means Clustering', "Selected": False})

    if alg == 'Agglomerative Clustering':
        alg_array.append({'name': 'Agglomerative Clustering', "Selected": True})
    else:
        alg_array.append({'name': 'Agglomerative Clustering', "Selected": False})

    if alg == 'Fuzzy C-means Clustering':
        alg_array.append({'name': 'Fuzzy C-means Clustering', "Selected": True})
    else:
        alg_array.append({'name': 'Fuzzy C-means Clustering', "Selected": False})

    if alg == 'Birch Clustering':
        alg_array.append({'name': 'Birch Clustering', "Selected": True})
    else:
        alg_array.append({'name': 'Birch Clustering', "Selected": False})

    return alg_array


def get_data_array(data):

    data_array = []

    if data == '50':
        data_array.append({'name': '50', "Selected": True})
    else:
        data_array.append({'name': '50', "Selected": False})

    if data == '100':
        data_array.append({'name': '100', "Selected": True})
    else:
        data_array.append({'name': '100', "Selected": False})

    if data == '250':
        data_array.append({'name': '250', "Selected": True})
    else:
        data_array.append({'name': '250', "Selected": False})

    if data == '400':
        data_array.append({'name': '400', "Selected": True})
    else:
        data_array.append({'name': '400', "Selected": False})

    if data == '500':
        data_array.append({'name': '500', "Selected": True})
    else:
        data_array.append({'name': '500', "Selected": False})

    if data == '1000':
        data_array.append({'name': '1000', "Selected": True})
    else:
        data_array.append({'name': '1000', "Selected": False})

    if data == '2500':
        data_array.append({'name': '2500', "Selected": True})
    else:
        data_array.append({'name': '2500', "Selected": False})

    return data_array


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
