from transformers import pipeline

classifier = pipeline(
    task="sentiment-analysis",
     model="./local_model"
)

result = classifier(
    "Kubernetes cluster performance is excellent today"
)

print(result)