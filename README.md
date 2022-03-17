# Automatic Image Tagging within Jina


## Step 1: Index Image Corpus Dataset

```bash
$ python mian.py index -f flow.yml -d /path/to/dataset
```

## Step 2: Start Jina Search Service

```bash
$ python main.py serve -f flow.yml
```

## Step 3: Predict tags for new image


For instance, you can follow the `example.py` to see how to predict:

```bash
$ python example.py
```
