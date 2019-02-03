# protein clustering webapp


- For running it on your local machine, you need to install python 3.6 and all the needed dependencies (sklearn)

    - after all the imports are finished, go to the /src folder and type the command: export FLASK_APP=app.py
    
    - then type: flask run
    
    - after that, you should see this in the command line: "* Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
    "
    - open a browser and access: http://127.0.0.1:5000/proteins
    
    
- For running the docker image:

    - after installing docker on your machine, run the following command
     
    - docker pull salbert/proteinclustering
    
    - type "docker build -f Dockerfile -t som:1.0 ." in the command line
    
    - after type: docker run  -p 5000:5000 som:1.0
    
    - after open a browser and access: http://0.0.0.0:5000/proteins
    
- The proteins page allows computing various clustering algorithms on the protein conformations (Structural Alphabet + RSA values)

- Similar to the work done here: https://www.sciencedirect.com/science/article/pii/S1877050918311797 

- The SOM section will display the analysis of proteins based on structure similar to the work done here:
http://www.cs.ubbcluj.ro/~studia-i/journal/journal/article/view/9/8 