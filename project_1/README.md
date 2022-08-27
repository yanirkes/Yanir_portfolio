# APP TREE
There are 4 ways to run the app:
1. run uvicorn RestAPI:app --reload in the terminal. Make sure that you are located at the primary directory

2. A local run by typing the following in the cmd:
*"docker-compose up"*. this will create a docker container that will contain the app.
   If using docker compass, you can easily can run the app by clicking on the browser button

3. Through the Kubernetes API. In order to activate the Kubernetes, please docker
as follows:
   * run the docker container as mentioned above.
   * Make sure you have downloaded kubernetes minikube (https://kubernetes.io/releases/download/)
   * open another terminal and run:
    1. minikube start
    2. kubectl apply -f ./deployment.yaml
    3. kubectl apply -f ./service.yaml
    4. minikube service <service name from the service file> (e.g, "tree-service")
   * you will be able to open browser and see the app
    
4. To run the app using KubeSail. To do that, please add to the previous step, the follows:
    * in the terminal:
    1. kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v0.48.1/deploy/static/provider/cloud/deploy.yaml
    2. minikube addons enable ingress
    3. kubectl apply -f ./ingress.yaml
    * create a user in https://docs.kubesail.com/    
    * run the command in the terminal (change the   USERNAME):  kubectl create -f https://byoc.kubesail.com/USERNAME.yaml
    * Copy the domain and run the app

