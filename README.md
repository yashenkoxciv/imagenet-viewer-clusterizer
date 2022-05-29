# imagenet-viewer-clusterizer

Do clustering for previously matched images.

# Requirements

1. Ubuntu 20.04.3
2. Python 3.8.10
3. requirements.txt
4. python -m pip install git+https://github.com/yashenkoxciv/imagenet-viewer.git


# Expected environment variables

| Name                    | Description                                                                         |
|-------------------------|:------------------------------------------------------------------------------------|
| RABBITMQ_HOST           | RabbitMQ's host                                                                     |
| INPUT_QUEUE             | RabbitMQ's queue with clustering request id                                         |
| OUTPUT_QUEUE            | RabbitMQ's queue to push found clusters (not implemented)                           |
| MATCHING_INPUT_QUEUE    | encoder's output queue; it is used to send unclusterized images to matching (again) |
| MONGODB_HOST            | MongoDB's connection string like this: mongodb://host:port/imagenetviewer           |
| CLUSTERING_THRESHOLD    | eps parameter for DBSCAN                                                            |
| CLUSTERING_MIN_CLUSTER  | Minimal size of a cluster                                                           |

