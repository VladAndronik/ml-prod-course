# Week 5

## Forward pass benchmark

```
python benchmark/benchmark_forward.py
```

```
Batch Size: 4 | Use Half: False | GPU Memory Init: 20.0 | GPU Memory overall: 40.0
Total speed: 26120.936025782747 FPS

Batch Size: 4 | Use Half: True | GPU Memory Init: 20.0 | GPU Memory overall: 40.0
Total speed: 24250.839813824405 FPS

```

## Locust benchmark
```
uvicorn --host 0.0.0.0 --port 8080 --workers 2 serving.fast_api:app
locust -f benchmark/locust_conf.py  --host http://54.221.129.217:7777 -r 1 -u 10
```

Ran 10 users at the same time.
```
Type     Name                                                                          # reqs      # fails |    Avg     Min     Max    Med |   req/s  failures/s
--------|----------------------------------------------------------------------------|-------|-------------|-------|-------|-------|-------|--------|-----------
POST     /predict                                                                       13493     0(0.00%) |     56      43     100     55 |  173.84        0.00
--------|----------------------------------------------------------------------------|-------|-------------|-------|-------|-------|-------|--------|-----------
         Aggregated                                                                     13493     0(0.00%) |     56      43     100     55 |  173.84        0.00
```

## FastAPI k8s deployment

```
docker build -f Dockerfile --tag 8092/app-fastapi --target app-fastapi .
docker push 8092/app-fastapi:latest
kubectl create -f k8s/app-fastapi.yml
kubectl port-forward --address 0.0.0.0 svc/app-fastapi 8080:8080
```

## Streamlit k8s deployment

```
docker build -f Dockerfile --tag 8092/app-streamlit --target app-streamlit .
docker push 8092/app-streamlit:latest
kubectl create -f k8s/app-streamlit.yml
kubectl port-forward --address 0.0.0.0 svc/app-streamlit 8080:8080
```

## Optimize

Baseline:
    Accuracy: 97.58
    Batch Size: 4 | Use Half: False | GPU Memory Init: 20.0 | GPU Memory overall: 40.0
    Total speed: 26120.936025782747 FPS
    Memory: 4.8MB

Static Quantization (only cpu):
    Accuracy: 96.8
    Batch Size: 4 | Use Half: False | GPU Memory Init: 0.0 | GPU Memory overall: 0.0
    Total speed: 8929.608320071535 FPS
    Memory: Size (MB): 1.206127

Quantization does not affect accuracy too much, but performance dips as it is processed on CPU only.


## Cloud Run
```
gcloud auth login
# create service account and generate key file for him. then this command.
gcloud auth activate-service-account test-account@helical-song-367521.iam.gserviceaccount.com --key-file helical-song-367521-e7c23ab29b30.json
gcloud auth configure-docker
docker push gcr.io/helical-song-367521/app-fastapi:latest
```


## K8S engine in GCP
```
https://cloud.google.com/kubernetes-engine/docs/deploy-app-cluster#dockerfile
https://cloud.google.com/blog/products/containers-kubernetes/kubectl-auth-changes-in-gke
https://cloud.google.com/sdk/docs/install#deb
https://cloud.google.com/artifact-registry/docs/docker/pushing-and-pulling
```

### Commands
```
sudo apt-get install apt-transport-https ca-certificates gnupg
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
sudo apt-get update && sudo apt-get install google-cloud-cli
sudo apt-get install google-cloud-sdk-gke-gcloud-auth-plugin

# create cluster
gcloud container clusters create-auto mnist-fastapi-cluster --region=us-west1
gcloud container clusters get-credentials mnist-fastapi-cluster --region us-west1

# push to artifact registry
gcloud auth configure-docker us-west1-docker.pkg.dev
docker tag 8092/app-fastapi:latest us-west1-docker.pkg.dev/helical-song-367521/mnist-artifacts/app-fastapi
gcloud auth configure-docker
docker push us-west1-docker.pkg.dev/helical-song-367521/mnist-artifacts/app-fastapi

# create deployment and service
kubectl create deployment app-fastapi --image us-west1-docker.pkg.dev/helical-song-367521/mnist-artifacts/app-fastapi:latest
kubectl expose deployment app-fastapi --type LoadBalancer --port 80 --target-port 8080
kubectl get service app-fastapi  # get external IP from it
```
