dlib-serving
============

```
docker run -it --rm yoanlin/dlib-serving:latest /shape-detection-server
```

Start the shape training:

```
docker run -it --rm \
    -v dlib/examples/faces:/data \
    yoanlin/dlib-serving:latest /train-shape-predictor --data-dir /data
```

## License

MIT License
