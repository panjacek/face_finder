IMAGES ?= $(PWD)
SERVICE_NAME ?= face_finder

ifeq ($(USE_GPU), 1)
	SERVICE_NAME += _gpu
endif

build:
	docker compose build

shell:
	docker compose run --rm $(SERVICE_NAME) /bin/bash

run:
	docker compose run --rm -e IMAGES=$(IMAGES) $(SERVICE_NAME)

format:
	isort .
	black .


up:
	docker compose up face_finder_web
	
down: 
	docker compose down --remove-orphans --volumes face_finder_web
