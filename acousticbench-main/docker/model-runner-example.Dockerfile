# Docker example: wrap a model so it can run isolated.
# Build: docker build -f docker/model-runner-example.Dockerfile -t echobench-model:example .
# The harness core never requires Docker. Pass --docker-image only when you want
# the run provenance to record docker isolation; inference still uses the local
# plugin contract unless you replace `echobench.engine` invocation with a
# container call (see ARCHITECTURE.md § Model isolation).
FROM python:3.11-slim
WORKDIR /model
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY user_models ./user_models
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir .
CMD ["echobench", "runs"]
