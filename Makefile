check-python:
	flake8
	isort --check-only
	black --check .
check: check-python

format:
	isort -y
	black .

test:
	python -m pytest -s \
		--cov=functions --cov-report=html --cov-report=term \
		--durations=0 $(FILTER)

docker-build:
	docker build -t rmeinl/python3.7-gensim:latest .

docker-push:
	docker push rmeinl/python3.7-gensim