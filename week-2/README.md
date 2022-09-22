# Week 2

```
kind create cluster --name week-2
```

## MINIO
```
kubectl create -f minio_prj/pv.yaml
kubectl create -f minio_prj/services.yaml
kubectl create -f minio_prj/deployment.yaml
```

### Run tests

```
kubectl port-forward svc/minio-service-api 9000:9000
python -m minio_prj.tests
```
