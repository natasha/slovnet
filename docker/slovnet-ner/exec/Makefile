
ROOT = ~/proj/slovnet
IMAGE = natasha/slovnet-ner

image:
	cd $(ROOT); docker build \
		-t $(IMAGE) \
		-f docker/slovnet-ner/exec/Dockerfile .

push:
	docker push $(IMAGE)

run:
	docker run -it --rm \
		-p 8080:8080 \
		-e BATCH_SIZE=4 \
		$(IMAGE)

