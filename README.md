## ML-FLOW WALKTHROUGH
1. Install the mlflow package
```
pip install mlflow
```
2. Structure the code by writing proper functions for different process
3. keep the <b>mlflow.start_run()</b> function and save the file as a '.py' script
4. Run the file

```
python mlflow_test.py
```

5. Start the mlflow UI
```
mlflow ui
```
6. Run the MLflow server 
```
mlflow server --backend-store-uri mlruns/ --default-artifact-root mlruns/ --host 0.0.0.0 --port 5000
```

<b>NOTE:<b>
<i> Data and Model (.pkl) files are not being uploaded due to the size issue.
