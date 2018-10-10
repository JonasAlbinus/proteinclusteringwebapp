import os
import threading

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

    if step == 2500:
        k = "4"
    elif step == 1000:
        k = "10"
    elif step == 500:
        k = "20"
    elif step == 100:
        k = "100"
    else:
        k = "200"

    if alg == "K-Means Clustering":
        filename = ("static/data/" + protein + "_" + k + "_k-means_pca" + ".png")
        file = protein + "_" + k + "_k-means_pca" + ".png"
        algName = "k-means"
    elif alg == "Agglomerative Clustering":
        filename = ("static/data/" + protein + "_" + k + "_agg_pca" + ".png")
        file = protein + "_" + k + "_k-_agg_pca" + ".png"
        algName = "agg"
    elif alg == "birch":
        filename = ("static/data/" + protein + "_" + k + "_birch_pca" + ".png")
        file = protein + "_" + k + "_k-_birch_pca" + ".png"
        algName = "birch"
    else:
        filename = ("static/data/" + protein + "_" + k + "_mb-k-means_pca" + ".png")
        file = protein + "_" + k + "_mb-k-means_pca" + ".png"
        algName = "mb-k-means"

    print(filename)
    my_file = Path(filename)
    if my_file.is_file():
        return render_template('proteins.html', alg=alg, protein=protein, cluster_no=step,
                               data=[{'name': '50'}, {'name': '100'}, {'name': '500'}, {'name': '1000'},
                                     {'name': '2500'}],
                               proteins=[{'name': '1GO1'}, {'name': '1JT8'}, {'name': '1L3P'}, {'name': '1P1L'}],
                               method=[{'name': 'K-Means Clustering'}, {'name': 'Agglomerative Clustering'},
                                       {'name': 'MiniBatchKMeans Clustering'}, {'name': 'Birch Clustering'}],
                               user_image=filename,
                               filename=file
                               )
    else:
        compute_thread = threading.Thread(target=compute_analysis, args=(protein, int(step), algName))
        compute_thread.start()
        return render_template('proteins.html', alg=alg, protein=protein, cluster_no=step,
                               data=[{'name': '50'}, {'name': '100'}, {'name': '500'}, {'name': '1000'},
                                     {'name': '2500'}],
                               proteins=[{'name': '1GO1'}, {'name': '1JT8'}, {'name': '1L3P'}, {'name': '1P1L'}],
                               method=[{'name': 'K-Means Clustering'}, {'name': 'Agglomerative Clustering'},
                                       {'name': 'MiniBatchKMeans Clustering'}, {'name': 'Birch Clustering'}],
                               user_image="static/Refresh.png"
                               )


@app.route('/proteins', methods=['GET', 'POST'])
def proteins():
    return render_template('proteins.html',
                           data=[{'name': '50'}, {'name': '100'}, {'name': '500'}, {'name': '1000'},
                                 {'name': '2500'}],
                           proteins=[{'name': '1GO1'}, {'name': '1JT8'}, {'name': '1L3P'}, {'name': '1P1L'}],
                           method=[{'name': 'K-Means Clustering'}, {'name': 'Agglomerative Clustering'},
                                   {'name': 'MiniBatchKMeans Clustering'}, {'name': 'Birch Clustering'}]
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
    # response.cache_control.no_store = True
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response


if __name__ == '__main__':
    app.run()
