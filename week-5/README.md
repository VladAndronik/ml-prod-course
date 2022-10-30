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
