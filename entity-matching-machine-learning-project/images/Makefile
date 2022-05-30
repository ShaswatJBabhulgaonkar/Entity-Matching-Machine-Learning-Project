AWS_ACCOUNT_ID ?= 554187193495
AWS_REGION ?= us-east-2
TAG ?= $(CIRCLE_BRANCH)-$(CIRCLE_BUILD_NUM)

IMAGE_NAME = h1-data-science-ml-em-core
IMAGE_NAME_W_TAG = $(IMAGE_NAME):$(TAG)
DOCKER_REPO = $(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com
IMAGE_URL = $(DOCKER_REPO)/$(IMAGE_NAME_W_TAG)

.PHONY: all

all: deploy

docker-build:
	aws ecr get-login-password --region $(AWS_REGION) | docker login --username AWS --password-stdin 711395599931.dkr.ecr.$(AWS_REGION).amazonaws.com
	docker build -t $(IMAGE_NAME_W_TAG) .

docker-tag: docker-build
	@echo '$(PARSED_TAG)'
	docker tag $(IMAGE_NAME_W_TAG) $(IMAGE_URL)

docker-push: docker-tag
	aws ecr get-login-password --region $(AWS_REGION) | docker login --username AWS --password-stdin $(DOCKER_REPO)
	docker push $(IMAGE_URL)

build: docker-build
	@echo "Built $(IMAGE_NAME)"

deploy: docker-build docker-push
	@echo "Deployed $(IMAGE_NAME)"