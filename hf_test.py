from transformers import pipeline

classifier = pipeline(
    "sentiment-analysis"
)

result = classifier(
    "Kubernetes cluster performance is excellent today"
)

print(result)