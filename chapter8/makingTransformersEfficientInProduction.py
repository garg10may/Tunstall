from transformers import pipeline
from performanceBenchmark import PerformanceBenchmark
from datasets import load_dataset
from datasets import load_metric

bert_ckpt = "transformersbook/bert-base-uncased-finetuned-clinc"
pipe = pipeline("text-classification", model=bert_ckpt)

query = """Hey, I'd like to rent a vehicle from Nov 1st to Nov 15th in Paris and I need a 15 passenger van"""
print(pipe(query))

clinc = load_dataset("clinc_oos", "plus")
pb = PerformanceBenchmark(pipe, clinc["test"])
perf_metrics = pb.run_benchmark()
print(perf_metrics)

# accuracy_score = load_metric("accuracy")
