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

## Pandas profiling

```
docker build --tag pd_profiling:latest pd_data_check/
docker run -it --rm pd_profiling:latest
```

File format | Size | Load time | Save time
--- | --- | --- | --- |
csv | 0.9191207 | 0.0071675 | 0.061124563217163086
h5 | 0.3834228 | 0.0005700 | 0.0005168914794921875
npy | 0.3815917 | 0.0032780 | 0.055948734283447266
xr | 0.4675292 | 0.0231752 | 0.007308244705200195
parquet | 0.4689798 | 0.0071616 | 0.005130767822265625
