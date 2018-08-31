import os

from flask import Flask, flash, redirect, render_template, request, url_for
from clustering import cluster

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
    print(select)
    cluster(select)
    return render_template('clustering.html',
                           data=[{'name': '2'}, {'name': '3'}, {'name': '4'}, {'name': '5'}, {'name': '6'},
                                 {'name': '8'}],
                           user_image="static/plots/plot.png")


@app.route("/testproteins", methods=['GET', 'POST'])
def testproteins():
    cluster_no = request.form.get('comp_select')
    protein = request.form.get('protein_select')
    alg = request.form.get('alg_select')

    print("clusterNo " + cluster_no)
    print("protein " + protein)
    print("alg " + alg)
    if alg == "K-Means":
        filename = ("static/data/" + protein + "_" + cluster_no + "_k-means_pca" + ".jpg")
    else:
        filename = ("static/data/" + protein + "_" + cluster_no + "_agg_pca" + ".jpg")

    return render_template('proteins.html', alg=alg, protein=protein, cluster_no=cluster_no,
                           data=[{'name': '4'}, {'name': '10'}, {'name': '20'}, {'name': '100'}, {'name': '200'}],
                           proteins=[{'name': '1GO1'}, {'name': '1JT8'}, {'name': '1L3P'}, {'name': '1P1L'}],
                           method=[{'name': 'K-Means'}, {'name': 'Agglomerative'}],
                           user_image=filename
                           )


@app.route('/proteins', methods=['GET', 'POST'])
def proteins():
    return render_template('proteins.html',
                           data=[{'name': '4'}, {'name': '10'}, {'name': '20'}, {'name': '100'}, {'name': '200'}],
                           proteins=[{'name': '1GO1'}, {'name': '1JT8'}, {'name': '1L3P'}, {'name': '1P1L'}],
                           method=[{'name': 'K-Means'}, {'name': 'Agglomerative'}]
                           )


# No caching at all for API endpoints.
@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response
