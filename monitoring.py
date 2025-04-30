# monitoring.py
from prometheus_client import start_http_server, Counter, Histogram, Gauge
import time

class ModelMonitor:
    def __init__(self, port=8001):
        # Start Prometheus metrics server
        start_http_server(port=port)
        print(f"Prometheus metrics server started at http://localhost:{port}")
        
        # Define metrics
        self.request_counter = Counter(
            'fashion_classifier_requests_total', 
            'Number of prediction requests received',
            ['result']
        )
        
        self.prediction_latency = Histogram(
            'fashion_classifier_prediction_latency_seconds',
            'Time taken to make predictions',
            buckets=[0.05, 0.1, 0.2, 0.5, 1, 2, 5]
        )
        
        self.model_accuracy = Gauge(
            'fashion_classifier_model_accuracy_percent',
            'Current estimated model accuracy percentage'
        )
        
        # Set initial values
        self._update_model_accuracy()
    
    def record_prediction(self, result):
        """Record a prediction event"""
        self.request_counter.labels(result=result).inc()
    
    def record_latency(self, start_time, end_time=None):
        """Record prediction latency"""
        if end_time is None:
            end_time = time.time()
        latency = end_time - start_time
        self.prediction_latency.observe(latency)
    
    def _update_model_accuracy(self):
        """Load and update model accuracy from metadata"""
        try:
            import json
            with open("models/model_metadata.json", "r") as f:
                metadata = json.load(f)
                accuracy = metadata.get("accuracy", 0)
                self.model_accuracy.set(accuracy)
        except (FileNotFoundError, json.JSONDecodeError):
            self.model_accuracy.set(0)
    
    def update_accuracy(self, accuracy):
        """Update model accuracy after retraining"""
        self.model_accuracy.set(accuracy)
