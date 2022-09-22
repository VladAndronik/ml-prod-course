# Week 2

```
kind create cluster --name week-2
```

## MINIO
```
kubectl create -f minio/pv.yaml
kubectl create -f minio/services.yaml
kubectl create -f minio/deployment.yaml
```

Run tests

```
python -m minio_prj.tests
```


