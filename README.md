# PR1

```
docker build --tag app-server:latest ./app-server/
```

```
docker pull 8092/app-server:latest
docker run -it --rm -p 8000:8000 8092/app-server:latest
```
