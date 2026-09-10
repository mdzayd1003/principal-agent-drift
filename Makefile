.PHONY: install test quick run judge analyse clean

install:
	pip install -r requirements.txt

test:
	python -m pytest -q

# Full 45-episode grid against the offline behavioural stand-in, judged and analysed.
quick:
	python -m drift --results results/smoke quick

run:
	python -m drift --results results/run run --model $(MODEL)

judge:
	python -m drift --results results/run judge --model $(MODEL)

analyse:
	python -m drift --results results/run analyse

clean:
	rm -rf results/run .pytest_cache
